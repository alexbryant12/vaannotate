import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vaannotate.vaannotate_ai_backend.utils.jsonish import _jsonify_cols


def test_jsonify_cols_converts_structured_runs_to_strings():
    df = pd.DataFrame(
        {
            "llm_runs": [
                [{"a": 1}],
                ("tuple", {"nested": True}),
                {"already": "dict"},
                None,
                "existing-string",
                math.nan,
            ]
        }
    )

    out = _jsonify_cols(df, ["llm_runs"])

    assert not out["llm_runs"].apply(lambda v: isinstance(v, (list, tuple, dict, set))).any()
    assert out["llm_runs"].iloc[0].startswith("[")
    assert out["llm_runs"].iloc[1].startswith("[")
    assert out["llm_runs"].iloc[2].startswith("{")
