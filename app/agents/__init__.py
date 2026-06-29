__all__ = ["DiagnosisWorkflow", "create_diagnosis_graph"]


def __getattr__(name: str):
    if name in __all__:
        from app.agents.workflow import DiagnosisWorkflow, create_diagnosis_graph

        return {
            "DiagnosisWorkflow": DiagnosisWorkflow,
            "create_diagnosis_graph": create_diagnosis_graph,
        }[name]
    raise AttributeError(f"module 'app.agents' has no attribute {name!r}")
