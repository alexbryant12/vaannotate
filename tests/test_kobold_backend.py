from vaannotate.vaannotate_ai_backend.config import LLMConfig
from vaannotate.vaannotate_ai_backend.llm_backends import KoboldCppBackend, build_llm_backend


def test_build_kobold_backend_requires_loopback():
    cfg = LLMConfig(backend="koboldcpp", koboldcpp_endpoint="http://127.0.0.1:5001")
    backend = build_llm_backend(cfg)
    assert isinstance(backend, KoboldCppBackend)


def test_kobold_backend_rejects_non_localhost():
    cfg = LLMConfig(backend="koboldcpp", koboldcpp_endpoint="http://10.0.0.8:5001")
    try:
        build_llm_backend(cfg)
    except ValueError as exc:
        assert "localhost" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-localhost endpoint")
