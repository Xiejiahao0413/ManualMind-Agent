# Version History

This file records the local version tags visible in the repository. It does not include fabricated deployment or production metrics.

| Version tag | Commit | Capability | Validation result |
| --- | --- | --- | --- |
| `v0.1-local-demo` | `34ddbb4` | Local end-to-end multi-agent diagnosis demo with synthetic manuals | Local demo and pytest validation completed during development |
| `v0.2-streaming-safety` | `e516db5` | Streaming Output Guard with rolling buffer for cross-chunk sensitive data masking | Pytest validation completed during development |
| `v0.3-pdf-ingestion` | `8927dfd` | Text-based PDF manual parser, PDF indexing script, PDF diagnosis demo | Pytest and PDF demo validation completed during development |
| `v0.4-real-llm-report` | `c285499` | Optional real LLM report generation with template fallback | Pytest and demo scripts passed without requiring API key |
| `v0.5-milvus-retrieval` | `4c2d87a` | Optional Milvus/vector backend with memory fallback | Latest local validation before portfolio polish: 141 tests passed; demo/eval scripts passed |

## Notes

- The repository uses synthetic manuals and synthetic evaluation samples.
- The default local path does not require OpenAI credentials or a running Milvus service.
- Real LLM, real Milvus, OCR, persistent storage, and enterprise deployment are production extension points.
