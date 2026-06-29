from app.llm.base import BaseLLMClient, LLMReportContext, LLMReportResult


class MockLLMClient(BaseLLMClient):
    provider = "mock"

    def __init__(
        self,
        response_text: str = "",
        *,
        raise_error: bool = False,
        error_type: str = "mock_error",
    ) -> None:
        self.response_text = response_text
        self.raise_error = raise_error
        self.error_type = error_type

    async def generate_report(self, context: LLMReportContext) -> LLMReportResult:
        if self.raise_error:
            raise RuntimeError(self.error_type)
        if not self.response_text:
            return LLMReportResult(
                success=False,
                provider=self.provider,
                error_type="empty_output",
                fallback_required=True,
            )
        return LLMReportResult(success=True, text=self.response_text, provider=self.provider)
