import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.vectorstore.factory import create_vector_runtime
from scripts.vector_manuals import load_synthetic_manual_chunks


QUERY = "A100 E03 temperature sensor abnormal"
FILTERS = {"device_model": "A100"}


def main() -> None:
    corpus = load_synthetic_manual_chunks()
    runtime = create_vector_runtime()
    chunks = corpus["chunks"]
    error_type = None
    results = []
    try:
        embeddings = runtime.embedding_client.embed_texts([chunk.text for chunk in chunks])
        runtime.vector_store.upsert_chunks(chunks, embeddings)
        query_embedding = runtime.embedding_client.embed_text(QUERY)
        results = runtime.vector_store.search(query_embedding, top_k=5, filters=FILTERS)
    except Exception as exc:
        error_type = str(exc)

    print(
        json.dumps(
            {
                "vector_backend": runtime.vector_store.backend,
                "embedding_provider": runtime.embedding_client.provider,
                "fallback_used": runtime.fallback_used or runtime.status.fallback_used,
                "skipped_reason": runtime.status.skipped_reason,
                "error_type": error_type,
                "query": QUERY,
                "filters": FILTERS,
                "top_k_results": [
                    {
                        "chunk_id": result.chunk_id,
                        "score": result.score,
                        "source_refs": result.source_refs,
                        "device_model": result.device_model,
                        "fault_code": result.fault_code,
                        "page": result.page,
                        "source_file": result.source_file,
                        "content_type": result.content_type,
                        "text": result.text[:240],
                    }
                    for result in results
                ],
                "source_refs": sorted({source for result in results for source in result.source_refs}),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
