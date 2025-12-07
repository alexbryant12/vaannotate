from datetime import date, datetime
import sys
import types
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vaannotate.vaannotate_ai_backend.services.family_labeler import (
    FamilyLabeler,
    run_family_labeling_for_units,
)


def test_run_family_labeling_for_units_normalizes_date_predictions():
    class DummyFamilyLabeler:
        def __init__(self):
            self.cfg = types.SimpleNamespace(progress_min_interval_s=0.0)

        def label_family_for_unit(self, uid, label_types, per_label_rules, **kwargs):
            return [
                {
                    "unit_id": uid,
                    "label_id": "dob",
                    "prediction": date(2020, 1, 2),
                    "runs": [{"raw": {"reasoning": "chosen date"}}],
                    "consistency": 0.9,
                }
            ]

    fam = DummyFamilyLabeler()

    df = run_family_labeling_for_units(
        fam,
        unit_ids=["unit-1"],
        label_types={"dob": "date"},
        per_label_rules={"dob": "Provide date of birth"},
    )

    assert list(df.columns) == [
        "unit_id",
        "label_id",
        "llm_prediction",
        "llm_runs",
        "llm_consistency",
        "llm_reasoning",
    ]
    assert df.iloc[0].llm_prediction == "2020-01-02"
    assert df.iloc[0].llm_reasoning == "chosen date"


class _ProbeFamilyLabeler(FamilyLabeler):
    def __init__(self):
        # Bypass base initialization; only populate attributes needed by probe_units_label_tree.
        self.cfg = types.SimpleNamespace(progress_min_interval_s=0.0, n_probe_units=1)
        self.repo = types.SimpleNamespace(
            notes=pd.DataFrame({"unit_id": ["probe-1"]}),
        )

    def label_family_for_unit(self, uid, label_types, per_label_rules, **kwargs):
        return [
            {
                "unit_id": uid,
                "label_id": "signed",
                "prediction": datetime(2021, 5, 6, 12, 30, 0),
                "runs": [{"raw": {"reasoning": "observed signature date"}}],
                "consistency": 1.0,
            }
        ]


def test_probe_units_normalizes_date_predictions():
    fam = _ProbeFamilyLabeler()

    df = fam.probe_units_label_tree(
        enrich=False,
        label_types={"signed": "date"},
        per_label_rules={"signed": "Provide signature date"},
    )

    assert df.iloc[0].llm_prediction == "2021-05-06T12:30:00"
    assert df.iloc[0].llm_reasoning == "observed signature date"
