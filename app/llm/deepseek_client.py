from app.llm.base import BaseLLMClient, LLMReportContext, LLMReportResult


class DeepSeekReportClient(BaseLLMClient):
    provider = "deepseek"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = "https://api.deepseek.com",
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
                error_type="deepseek_sdk_not_installed",
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
        except Exception:
            return LLMReportResult(
                success=False,
                provider=self.provider,
                error_type="deepseek_api_error",
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


def _system_prompt(query_type: str | None) -> str:
    if query_type == "manual_qa":
        return (
            "你是设备手册问答助手。只能基于用户提供的 evidence 回答，不允许编造手册中没有的步骤、"
            "文件名或页码。如果 evidence 不足，必须明确回答“未在上传手册中找到依据”。"
            "回答应偏操作指南，包含操作步骤和引用来源。引用来源只能使用 source_refs 中给出的值。"
        )
    return (
        "你是设备故障诊断助手。只能基于用户提供的 evidence 和 safety_warnings 生成中文诊断报告，"
        "不允许编造手册中没有的原因、步骤、文件名或页码。报告必须包含：故障识别、可能原因、"
        "排查步骤、安全提醒、引用来源。如果 evidence 不足，必须明确回答“未在上传手册中找到依据”。"
        "引用来源只能使用 source_refs 中给出的值。"
    )


def _build_prompt(context: LLMReportContext) -> str:
    lines = [
        f"query_type: {context.query_type}",
        f"query: {context.query}",
        f"device_model: {context.device_model or ''}",
        f"fault_code: {context.fault_code or ''}",
        f"risk_level: {context.risk_level}",
        "evidence:",
    ]
    if context.evidence_items:
        for index, item in enumerate(context.evidence_items, start=1):
            lines.extend(
                [
                    f"[{index}] source_ref: {item.get('source_ref', '')}",
                    f"[{index}] section_title: {item.get('section_title', '')}",
                    f"[{index}] page: {item.get('page', '')}",
                    f"[{index}] text: {item.get('text', '')}",
                ]
            )
    else:
        lines.extend(f"- {item}" for item in context.evidence_snippets)
    lines.extend(
        [
            "safety_warnings:",
            *[f"- {item}" for item in context.safety_warnings],
            "source_refs:",
            *[f"- {item}" for item in context.source_refs],
            "handoff_required: " + str(context.handoff_required),
        ]
    )
    return "\n".join(lines)
