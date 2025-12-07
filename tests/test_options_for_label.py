from vaannotate.vaannotate_ai_backend.services.context_builder import (
    _options_for_label as context_options_for_label,
)
from vaannotate.vaannotate_ai_backend.services.rag_retriever import (
    _options_for_label as rag_options_for_label,
)


def test_context_builder_options_for_categorical_multi():
    cfg = {"Multi": {"options": ["A", "B"]}}

    assert context_options_for_label("Multi", "categorical_multi", cfg) == ["A", "B"]


def test_rag_retriever_options_for_categorical_multi():
    cfg = {"Multi": {"options": ["A", "B", "C"]}}

    assert rag_options_for_label("Multi", "categorical_multi", cfg) == ["A", "B", "C"]
