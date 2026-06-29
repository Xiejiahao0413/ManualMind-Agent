import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.vectorstore.factory import create_vector_runtime
from scripts.vector_manuals import load_synthetic_manual_chunks


def main() -> None:
    corpus = load_synthetic_manual_chunks()
    runtime = create_vector_runtime()
    chunks = corpus["chunks"]
    indexed_chunks_count = 0
    error_type = None
    try:
        embeddings = runtime.embedding_client.embed_texts([chunk.text for chunk in chunks])
        runtime.vector_store.upsert_chunks(chunks, embeddings)
        indexed_chunks_count = len(chunks)
    except Exception as exc:
        error_type = str(exc)
        indexed_chunks_count = 0

    print(
        json.dumps(
            {
                "vector_backend": runtime.vector_store.backend,
                "embedding_provider": runtime.embedding_client.provider,
                "indexed_chunks_count": indexed_chunks_count,
                "fallback_used": runtime.fallback_used or runtime.status.fallback_used,
                "skipped_reason": runtime.status.skipped_reason,
                "error_type": error_type,
                "source_files": corpus["source_files"],
                "device_models": corpus["device_models"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
