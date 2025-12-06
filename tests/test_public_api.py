from vaannotate.vaannotate_ai_backend import build_next_batch, run_inference


def test_run_inference_exposed_via_package_init():
    # The package should re-export inference-only orchestration alongside
    # active-learning helpers.
    from vaannotate.vaannotate_ai_backend import orchestrator

    assert run_inference is orchestrator.run_inference
    assert build_next_batch is orchestrator.build_next_batch
