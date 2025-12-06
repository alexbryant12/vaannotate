"""Uncertainty scoring for LLM probe results."""

from __future__ import annotations

import numpy as np
import pandas as pd


class LLMUncertaintyScorer:
    """Compute uncertainty/easiness signals for LLM-first probes."""

    def __init__(self, llm_first_config):
        self.cfg = llm_first_config

    def score_probe_results(self, probe_df: pd.DataFrame) -> pd.DataFrame:
        """Attach uncertainty scores/flags expected by downstream selection.

        The scoring mirrors the inline logic previously in ``engine.py``:
        - Prefer forced-choice entropy when available (already stored as ``U``).
        - Fall back to ``1 - consistency`` when forced-choice data is absent.
        - Ensure the ``U`` column exists for downstream selectors.
        """

        if probe_df is None or not isinstance(probe_df, pd.DataFrame):
            return pd.DataFrame()

        df = probe_df.copy()

        if "U" not in df.columns:
            df["U"] = np.nan

        if "fc_entropy" in df.columns:
            idx = df["U"].isna()
            if "consistency" in df.columns:
                df.loc[idx, "U"] = 1.0 - pd.to_numeric(
                    df.loc[idx, "consistency"], errors="coerce"
                ).fillna(0.0)
        elif "consistency" in df.columns:
            df["U"] = 1.0 - pd.to_numeric(df["consistency"], errors="coerce").fillna(0.0)

        return df


__all__ = ["LLMUncertaintyScorer"]
