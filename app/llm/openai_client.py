from app.llm.base import BaseLLMClient, LLMReportContext, LLMReportResult


class OpenAIReportClient(BaseLLMClient):
    provider = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        timeout_seconds: float = 20.0,
        max_tokens: int = 900,
        temperature: float = 0.2,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def generate_report(self, context: LLMReportContext) -> LLMReportResult:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            return LLMReportResult(
                success=False,
                provider=self.provider,
                error_type="openai_sdk_not_installed",
                fallback_required=True,
            )

        try:
            client = AsyncOpenAI(api_key=self.api_key, timeout=self.timeout_seconds)
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You generate concise Chinese equipment fault diagnosis reports. "
                            "Use only the provided sanitized context. Keep the exact sections: "
                            "故障识别, 可能原因, 排查步骤, 安全提醒, 引用来源."
                        ),
                    },
                    {"role": "user", "content": _build_prompt(context)},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            text = response.choices[0].message.content or ""
        except Exception:
            return LLMReportResult(
                success=False,
                provider=self.provider,
                error_type="openai_api_error",
                fallback_required=True,
            )

        if not text.strip():
            return LLMReportResult(
                success=False,
                provider=self.provider,
                error_type="empty_output",
                fallback_required=True,
            )
        return LLMReportResult(success=True, text=text.strip(), provider=self.provider)


def _build_prompt(context: LLMReportContext) -> str:
    return "\n".join(
        [
            f"query: {context.query}",
            f"device_model: {context.device_model or ''}",
            f"fault_code: {context.fault_code or ''}",
            f"risk_level: {context.risk_level}",
            "evidence_snippets:",
            *[f"- {item}" for item in context.evidence_snippets],
            "safety_warnings:",
            *[f"- {item}" for item in context.safety_warnings],
            "source_refs:",
            *[f"- {item}" for item in context.source_refs],
            "handoff_required: " + str(context.handoff_required),
        ]
    )
