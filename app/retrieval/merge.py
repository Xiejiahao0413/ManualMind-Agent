from app.schemas.retrieval import RetrievalResult


def merge_retrieval_results(
    bm25_results: list[RetrievalResult],
    dense_results: list[RetrievalResult],
) -> list[RetrievalResult]:
    merged: dict[str, RetrievalResult] = {}
    for result in bm25_results:
        merged[result.chunk_id] = result.model_copy(
            update={
                "source": "bm25",
                "bm25_score": result.bm25_score if result.bm25_score is not None else result.score,
            }
        )

    for result in dense_results:
        existing = merged.get(result.chunk_id)
        dense_score = result.dense_score if result.dense_score is not None else result.score
        if existing is None:
            merged[result.chunk_id] = result.model_copy(
                update={
                    "source": "dense",
                    "dense_score": dense_score,
                }
            )
            continue

        bm25_score = existing.bm25_score or 0.0
        fused_score = bm25_score + dense_score
        source_refs = sorted(set(existing.source_refs + result.source_refs))
        merged[result.chunk_id] = existing.model_copy(
            update={
                "score": fused_score,
                "bm25_score": bm25_score,
                "dense_score": dense_score,
                "source": "hybrid",
                "source_refs": source_refs,
            }
        )

    return sorted(merged.values(), key=lambda item: item.score, reverse=True)
