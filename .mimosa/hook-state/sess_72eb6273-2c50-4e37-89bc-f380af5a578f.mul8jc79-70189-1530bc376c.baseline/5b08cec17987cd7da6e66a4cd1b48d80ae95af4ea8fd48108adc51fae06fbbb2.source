# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DeepSeek official platform provider (OpenAI-compatible API).

Minimal subclass of the stock ``OpenAIProvider``: only credentials/endpoint
resolution and the output-ceiling policy differ from the parent class.

Credentials are read from the environment only, never hardcoded:
  DEEPSEEK_API_KEY    - API key (required)
  DEEPSEEK_BASE_URL   - endpoint override (default: https://api.deepseek.com/v1)

Model specs (platform datasheets, Sep 2026):
  deepseek-flash  = DeepSeek V4.1 Flash - 1M-token context, 384K-token max
                     output, supports thinking/reasoning mode (thinking tokens
                     inside completions are normal)
  deepseek-v4-pro = DeepSeek V4 Pro tier
The platform rejects any other model name with HTTP 400.

Output ceiling: no ``max_tokens`` is sent at all, so the API applies the
model's real maximum (384K for deepseek-flash) - any value injected here
could only truncate long kernel rewrites.
"""

import os
from typing import Any

from .openai_base import OpenAICompatibleProvider
from .openai_provider import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek chat models via the official OpenAI-compatible platform API."""

    def __init__(self):
        # Bypass OpenAIProvider.__init__ (it pins api_key_env="OPENAI_API_KEY"
        # and no base_url); configure the DeepSeek endpoint instead.
        OpenAICompatibleProvider.__init__(
            self,
            api_key_env="DEEPSEEK_API_KEY",
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        )

    @property
    def name(self) -> str:
        return "DeepSeek"

    def get_max_tokens_limit(self, model_name: str) -> int:
        # deepseek-flash: 1M context / 384K max output (platform datasheet).
        # Only used to pass the stock min() clamp; no cap is actually sent.
        return 393216

    def _build_api_params(
        self, model_name: str, messages: list[dict[str, str]], **kwargs
    ) -> dict[str, Any]:
        params = super()._build_api_params(model_name, messages, **kwargs)
        # Send no output cap: the platform enforces the model's real ceiling.
        params.pop("max_tokens", None)
        params.pop("max_completion_tokens", None)
        return params
