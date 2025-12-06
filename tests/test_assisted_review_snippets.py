from types import SimpleNamespace
from pathlib import Path

import pandas as pd

from vaannotate.rounds import AssignmentUnit, RoundBuilder
import vaannotate.rounds as rounds


def test_generate_assisted_review_snippets_smoke(monkeypatch, tmp_path: Path) -> None:
    builder = RoundBuilder(tmp_path)
    pheno_row = {"level": "multi_doc"}
    labelset = {"labelset_id": "ls1", "labels": [{"id": "lab1", "type": "text"}]}
    reviewer_assignments = {
        "reviewer": [
            AssignmentUnit(
                unit_id="u1",
                patient_icn="p1",
                doc_id="d1",
                payload={"documents": [{"doc_id": "d1", "text": "note text"}]},
            )
        ]
    }
    config = {"assisted_review": {"enabled": True, "top_snippets": 1}, "ai_backend": {}}

    # Avoid optional parquet dependencies during the smoke test
    monkeypatch.setattr(
        pd.DataFrame, "to_parquet", lambda self, path, index=False: Path(path).write_text("stub"), raising=False
    )

    class DummyConfig:
        def __init__(self) -> None:
            self.final_llm_labeling = False
            self.llm = SimpleNamespace()
            self.llmfirst = SimpleNamespace(
                single_doc_context="rag", single_doc_full_context_max_chars=None
            )
            self.rag = None
            self.index = None
            self.scjitter = None

    class DummyPaths:
        def __init__(self, *args: object) -> None:
            self.args = args

    class DummyRunner:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.store = SimpleNamespace(build_chunk_index=lambda *a, **kw: None)
            self.repo = SimpleNamespace(notes=None)
            self.cfg = SimpleNamespace(
                rag=None,
                index=None,
                llmfirst=SimpleNamespace(
                    single_doc_context="rag", single_doc_full_context_max_chars=None
                ),
                scjitter=None,
            )
            self.llm = object()
            self.retriever = object()
            self.label_config = object()

        def _label_maps(self):
            return {}, {}, {"lab1": "rule"}, {"lab1": "text"}

    class DummyFamilyLabeler:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def ensure_label_exemplars(self, *args: object, **kwargs: object) -> None:
            return None

    def dummy_contexts(*args: object, **kwargs: object):
        return [
            {
                "doc_id": "d1",
                "chunk_id": 0,
                "score": 0.9,
                "source": "unit",
                "text": "context text",
                "metadata": {"foo": "bar"},
            }
        ]

    import sys
    import types

    config_module = types.SimpleNamespace(OrchestratorConfig=DummyConfig, Paths=DummyPaths)
    label_configs_module = types.SimpleNamespace(LabelConfigBundle=lambda **kwargs: kwargs)
    orchestration_module = types.SimpleNamespace(
        build_active_learning_runner=lambda **_: DummyRunner()
    )
    contexts_module = types.SimpleNamespace(_contexts_for_unit_label=dummy_contexts)
    family_labeler_module = types.SimpleNamespace(FamilyLabeler=DummyFamilyLabeler)

    monkeypatch.setitem(sys.modules, "vaannotate.vaannotate_ai_backend.config", config_module)
    monkeypatch.setitem(sys.modules, "vaannotate.vaannotate_ai_backend.label_configs", label_configs_module)
    monkeypatch.setitem(sys.modules, "vaannotate.vaannotate_ai_backend.orchestration", orchestration_module)
    monkeypatch.setitem(sys.modules, "vaannotate.vaannotate_ai_backend.services.contexts", contexts_module)
    monkeypatch.setitem(sys.modules, "vaannotate.vaannotate_ai_backend.services.family_labeler", family_labeler_module)

    result = builder._generate_assisted_review_snippets(
        pheno_row=pheno_row,
        labelset=labelset,
        round_dir=tmp_path,
        reviewer_assignments=reviewer_assignments,
        config=config,
        config_base=tmp_path,
        top_n=1,
    )

    assert result["unit_snippets"]["u1"]["lab1"][0]["text"] == "context text"
