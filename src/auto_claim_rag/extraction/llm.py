from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI


class LLM:
    def complete(self, prompt: str) -> str:
        raise NotImplementedError


@dataclass
class OpenAIChatLLM(LLM):
    api_key: str
    model: str
    base_url: str | None = None
    temperature: float = 0.0
    json_mode: bool = True

    def __post_init__(self) -> None:
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def complete(self, prompt: str) -> str:
        kwargs = {}
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a careful information extraction system.",
                    },
                    {"role": "user", "content": prompt},
                ],
                **kwargs,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            resp = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a careful information extraction system.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return (resp.choices[0].message.content or "").strip()


@dataclass
class StaticLLM(LLM):
    content: str

    def complete(self, prompt: str) -> str:
        return self.content
