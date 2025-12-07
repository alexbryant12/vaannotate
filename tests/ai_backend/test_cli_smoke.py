from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "ai_backend"


@pytest.mark.parametrize("with_label_config", [True, False])
def test_cli_active_learning_smoke(tmp_path: Path, with_label_config: bool) -> None:
    env = os.environ.copy()
    env["VAANNOTATE_TEST_STUB_ACTIVE_RUNNER"] = "1"

    if with_label_config:
        label_config_flag = ["--label-config", str(DATA_DIR / "label_config.json")]
    else:
        label_config_flag = []

    cmd = [
        sys.executable,
        "-m",
        "vaannotate.vaannotate_ai_backend.cli",
        "--notes",
        str(DATA_DIR / "notes.csv"),
        "--annotations",
        str(DATA_DIR / "annotations.csv"),
        "--outdir",
        str(tmp_path),
        *label_config_flag,
    ]

    result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)

    csv_path = tmp_path / "ai_next_batch.csv"
    assert csv_path.exists()

    df = pd.read_csv(csv_path)
    assert not df.empty
    assert set(df["unit_id"].astype(str).unique()) >= {"1001", "1002"}

    stdout = result.stdout.strip().splitlines()
    assert any("ai_next_batch" in line for line in stdout)


def test_cli_active_learning_outputs(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["VAANNOTATE_TEST_STUB_ACTIVE_RUNNER"] = "1"

    cmd = [
        sys.executable,
        "-m",
        "vaannotate.vaannotate_ai_backend.cli",
        "--notes",
        str(DATA_DIR / "notes.csv"),
        "--annotations",
        str(DATA_DIR / "annotations.csv"),
        "--label-config",
        str(DATA_DIR / "label_config.json"),
        "--outdir",
        str(tmp_path),
    ]

    subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)

    expected = {
        "ai_next_batch.csv",
        "bucket_disagreement.parquet",
        "bucket_llm_uncertain.parquet",
        "bucket_llm_certain.parquet",
        "bucket_diversity.parquet",
    }

    for name in expected:
        assert (tmp_path / name).exists()

    df = pd.read_csv(tmp_path / "ai_next_batch.csv")
    assert "selection_reason" in df.columns

