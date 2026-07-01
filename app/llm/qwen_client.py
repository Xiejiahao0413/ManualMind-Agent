from app.llm.base import BaseLLMClient, LLMReportContext, LLMReportResult
from app.llm.deepseek_client import _build_prompt, _system_prompt


class QwenReportClient(BaseLLMClient):
    provider = "qwen"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout_seconds: float = 20.0,
        max_tokens: int = 900,
        temperature: float = 0.2,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries

    async def generate_report(self, context: LLMReportContext) -> LLMReportResult:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            return LLMReportResult(
                success=False,
                provider=self.provider,
                model=self.model,
                error_type="qwen_sdk_not_installed",
                fallback_required=True,
            )

        try:
            client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                max_retries=self.max_retries,
            )
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": _system_prompt(context.query_type),
                    },
                    {"role": "user", "content": _build_prompt(context)},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            text = response.choices[0].message.content or ""
        except Exception as exc:
            return LLMReportResult(
                success=False,
                provider=self.provider,
                model=self.model,
                error_type=f"qwen_{type(exc).__name__}",
                error_message=str(exc),
                fallback_required=True,
            )

        if not text.strip():
            return LLMReportResult(
                success=False,
                provider=self.provider,
                model=self.model,
                error_type="qwen_empty_output",
                fallback_required=True,
            )
        return LLMReportResult(success=True, text=text.strip(), provider=self.provider, model=self.model)
