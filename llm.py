from __future__ import annotations
import json
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from engine.utils.cache import Cache
from helpers.loggers import CustomLogger
from engine.utils.monitoring import CostEstimator  # Ensure this is correctly imported
import tiktoken
import threading
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Optional, List
import requests

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult, BaseMessage

class CostTrackingCallback(BaseCallbackHandler):
    def __init__(self, manager):
        self.manager = manager
        self._last_messages: List[BaseMessage] | List[str] = []

    def on_llm_start(
        self,
        serialized: dict,
        messages: Optional[List[BaseMessage]] = None,
        prompts: Optional[List[str]] = None,
        **kwargs,
    ):
        # Grab whichever the LLM is sending us: full message objects or raw prompt-strings
        self._last_messages = messages or prompts or []

    def on_llm_end(self, response: LLMResult, **kwargs):
        output_text_parts = []
        for generation in response.generations:
            for gen in generation:
                # Prioritize content if exists
                if gen.text:
                    output_text_parts.append(gen.text)
                # Include tool calls as JSON strings to count tokens properly
                elif gen.message and gen.message.additional_kwargs.get('tool_calls'):
                    tool_calls = gen.message.additional_kwargs['tool_calls']
                    tool_calls_text = json.dumps(tool_calls)
                    output_text_parts.append(tool_calls_text)

        output_text = "".join(output_text_parts)

        # Log explicitly if still empty (unlikely after above logic)
        if not output_text.strip():
            self.manager.logger.warning(f"Truly empty LLM response: {response.generations}")

        self.manager._record_cost(self._last_messages, output_text)

class LLMManager:
    """
    Handles LLM initialization and usage independently of caching.
    Integrates cost estimation for better cost tracking.
    Also enforces a 'max_tokens' limit for each model, if specified.
    """

    # ------------------------------------------------------------------
    #  LATEST PRICES – 9 May 2025  |  All values are **USD per 1 M tokens**
    # ------------------------------------------------------------------
    MODEL = {
        # ---------- OpenAI ----------
        "gpt-5":            {"input_cost": 1.25,   "output_cost":  10.00, "max_tokens": 1000000},
        "gpt-5-mini":       {"input_cost": 0.25,   "output_cost":  2.00, "max_tokens": 256000},
        "gpt-4.1":          {"input_cost": 2.00,   "output_cost":  8.00, "max_tokens": 1000000},
        "gpt-4.1-mini":     {"input_cost": 0.40,   "output_cost":  1.60, "max_tokens": 1000000},
        "gpt-4.1-nano":     {"input_cost": 0.10,   "output_cost":  0.40, "max_tokens": 1000000},
        "gpt-4o":           {"input_cost": 2.50,   "output_cost": 10.00, "max_tokens": 128000},
        "gpt-4o-mini":      {"input_cost": 0.15,   "output_cost":  0.60, "max_tokens": 128000},
        "o3":               {"input_cost":10.00,   "output_cost": 40.00, "max_tokens": 200000},
        "o4-mini":          {"input_cost": 1.10,   "output_cost":  4.40, "max_tokens": 200000},
        # legacy keys (kept for cache compatibility)
        "o1":               {"input_cost":15.00,   "output_cost": 60.00, "max_tokens": 128000},
        "o1-preview":       {"input_cost":15.00,   "output_cost": 60.00, "max_tokens": 128000},
        "o1-mini":          {"input_cost": 3.00,   "output_cost": 12.00, "max_tokens":  32000},
        "gpt-4-turbo":      {"input_cost":10.00,   "output_cost": 30.00, "max_tokens":  32000},
        "gpt-3.5-turbo":    {"input_cost": 0.50,   "output_cost":  1.50, "max_tokens":  16000},

        # ---------- Anthropic ----------
        "claude-3.7-sonnet":{"input_cost": 3.00,   "output_cost": 15.00, "max_tokens": 200000},
        "claude-3.5-sonnet":{"input_cost": 3.00,   "output_cost": 15.00, "max_tokens": 200000},
        "claude-3.5-haiku": {"input_cost": 0.80,   "output_cost":  4.00, "max_tokens": 200000},
        "claude-3-opus":    {"input_cost":15.00,   "output_cost": 75.00, "max_tokens": 200000},

        # ---------- Google Gemini ----------
        "gemini-2.5-pro":   {"input_cost": 1.25,   "output_cost": 10.00, "max_tokens": 200000},
        "gemini-2.0-flash": {"input_cost": 0.10,   "output_cost":  0.40, "max_tokens": 1000000},
        "gemini-2.0-flash-lite":{"input_cost":0.019,"output_cost": 0.019,"max_tokens": 1000000},
        "gemini-1.5-pro":   {"input_cost": 1.25,   "output_cost":  5.00, "max_tokens": 2000000},
        "gemini-1.5-flash": {"input_cost": 0.075,  "output_cost":  0.30, "max_tokens": 1000000},

        # ---------- Amazon Nova ----------
        "amazon-nova-micro":{"input_cost": 0.035,  "output_cost":  0.14, "max_tokens":   4096},
        "amazon-nova-lite": {"input_cost": 0.060,  "output_cost":  0.24, "max_tokens":   8192},
        "amazon-nova-pro":  {"input_cost": 0.800,  "output_cost":  3.20, "max_tokens":  32000},

        # ---------- Cohere ----------
        "command-a":        {"input_cost": 2.50,   "output_cost": 10.00, "max_tokens": 128000},
        "command-r-plus":   {"input_cost": 2.50,   "output_cost": 10.00, "max_tokens":  32000},
        "command-r":        {"input_cost": 0.15,   "output_cost":  0.60, "max_tokens":   8192},
        "command-r7b":      {"input_cost": 0.0375, "output_cost":  0.15, "max_tokens": 128000},

        # ---------- Mistral AI ----------
        "mistral-large":    {"input_cost": 2.00,   "output_cost":  6.00, "max_tokens":  32000},
        "mistral-small":    {"input_cost": 0.20,   "output_cost":  0.60, "max_tokens":  32000},
        "mistral-nemo":     {"input_cost": 0.15,   "output_cost":  0.15, "max_tokens":  32000},
        "pixtral-12b":      {"input_cost": 0.15,   "output_cost":  0.15, "max_tokens":  32000},
        "mistral-8b":     {"input_cost": 0.07,   "output_cost":  0.21, "max_tokens":  16000},
        "mistral-3b":     {"input_cost": 0.02,   "output_cost":  0.06, "max_tokens":   8000},

        # ---------- Meta (Llama 3 via Deepinfra) ----------
        "llama-3-1-70b":    {"input_cost": 0.23,   "output_cost":  0.40, "max_tokens":  32000},

        # ---------- DeepSeek ----------
        "deepseek-v3":      {"input_cost": 0.14,   "output_cost":  0.28, "max_tokens":  32000},
    }

    _DEC6 = Decimal("0.000001")

    def __init__(self, model_name: str, cache: Optional[Cache] = None, temperature: float = 0.2):
        self.logger         = CustomLogger("LLM Manager").get_logger()
        self.model_name     = model_name
        self.cache          = cache
        self.temperature    = temperature

        if model_name not in self.MODEL:
            raise ValueError(f"Unknown model '{model_name}'. Add it to MODEL first.")

        # 1) Build real provider
        self.llm = self._init_provider()

        pricing_map = self._fetch_model_pricing()
        # 2) Prepare cost estimator
        meta = pricing_map[model_name]
        self.max_tokens     = meta["max_tokens"]
        self.cost_estimator = CostEstimator(
            model       = model_name,
            input_cost  = meta["input_cost"],
            output_cost = meta["output_cost"],
        )
        self._last_cost: dict | None = None

    def _init_provider(self):
        cb = CostTrackingCallback(self)
        if self.model_name.startswith(("gpt-", "o4", "o3")):
            return ChatOpenAI(model=self.model_name, streaming=True, callbacks=[cb], temperature=self.temperature)
        if "claude" in self.model_name:
            return ChatAnthropic(model=self.model_name, streaming=True, callbacks=[cb], temperature=self.temperature)
        raise ValueError(f"Unsupported model {self.model_name}")

    # -------------------------------------------------------------------------
    #  Internal: invoked by our proxy on every model call
    # -------------------------------------------------------------------------
    def _record_cost(self, messages: list[str] | str, output_text: str):
        """
        Record cost using either a single string or a list of prompt-strings.
        """
        # Build a single input string from whatever we got
        if isinstance(messages, str):
            inp_text = messages
            if not inp_text or not output_text:
                self.logger.warning("Cost recording skipped due to missing input/output")
                return
        else:
            # list of plain strings → just join
            if all(isinstance(m, str) for m in messages):
                inp_text = " ".join(messages)
            # (fallback) list of message‐objects with .content
            else:
                inp_text = " ".join(
                    m.content for m in messages
                    if hasattr(m, "content") and isinstance(m.content, str)
                )

        # Skip if empty
        self.logger.debug(f"_record_cost: inp_text='{inp_text}', output_text='{output_text}'")
        if not inp_text or not output_text:
            self.logger.warning("Cost recording skipped due to missing input/output")
            return

        # Compute and stash
        self._last_cost = self.cost_estimator.calculate_cost(inp_text, output_text)
        self.logger.debug(f"Cost recorded: {self._last_cost}")


    def _fetch_model_pricing(self):
        OPENROUTER_API_KEY = "your_api_key"
        HEADERS = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "your-app-url",   
        }

        url = "https://openrouter.ai/api/v1/models"
        response = requests.get(url, headers=HEADERS)
        
        if response.status_code != 200:
            raise Exception(f"Failed to fetch model metadata: {response.text}")
        
        data = response.json()["data"]

        pricing_map = {}
        for model in data:
            model_id = model["id"]
            model_name = model.get("name", "")
            pricing = model.get("pricing", {})
            
            pricing_map[model_id] = {
                "model_name": model_name,
                "input_cost": float(pricing.get("prompt", 0)*LLMManager._DEC6),      
                "output_cost": float(pricing.get("completion", 0)*LLMManager._DEC6), 
                "max_tokens": model.get("context_length", 0)
            }
        
        return pricing_map

    # -------------------------------------------------------------------------
    #  Public helpers
    # -------------------------------------------------------------------------
    def get_llm(self) -> Any:
        return self.llm

    def get_last_cost_breakdown(self) -> dict:
        """Cost details (input_tokens, input_cost, ... total_cost, cumulative_cost)."""
        return self._last_cost or {
            'input_tokens': 0,
            'output_tokens': 0,
            'input_cost': 0.0,
            'output_cost': 0.0,
            'total_cost': 0.0,
            'cumulative_cost': self.get_total_cost()
        }

    def get_total_cost(self) -> float:
        """Cumulative spend on this manager."""
        return self.cost_estimator.get_total_cost()

    # -------------------------------------------------------------------------
    #  Token truncation helper
    # -------------------------------------------------------------------------
    def truncate_tokens(
        self,
        text: str,
        custom_max_tokens: Optional[int] = None,
        model_override: Optional[str] = None,
    ) -> str:
        model_for_encoding = model_override or self.model_name
        limit = custom_max_tokens or self.max_tokens

        try:
            enc = tiktoken.encoding_for_model(model_for_encoding)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        toks = enc.encode(text)
        if len(toks) <= limit:
            return text

        self.logger.warning(f"Truncating from {len(toks)}→{limit} tokens")
        return enc.decode(toks[:limit])

    @staticmethod
    def get_model_encoder(model_name: str = None):
        try:
            tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # For newer models not yet in tiktoken
            if "4.1" in model_name:  # For GPT-4.1 family models
                tokenizer = tiktoken.get_encoding("cl100k_base")  # Use same encoding as GPT-4
            elif "gpt-4" in model_name or "o" in model_name:  # For other GPT-4 or OpenAI models
                tokenizer = tiktoken.get_encoding("cl100k_base")
            else:
                # Default fallback
                tokenizer = tiktoken.get_encoding("p50k_base")
            
        return tokenizer
