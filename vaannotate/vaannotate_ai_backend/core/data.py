from __future__ import annotations

import json
import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s).strip()


def safe_json_loads(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, (dict, list)):
        return x
    try:
        return json.loads(x)
    except Exception:
        try:
            import ast

            return ast.literal_eval(x)
        except Exception:
            return None


class DataRepository:
    """Canonical interface to notes, annotations, and labeling metadata."""

    def __init__(
        self,
        notes_df: pd.DataFrame,
        ann_df: pd.DataFrame,
        *,
        phenotype_level: str | None = None,
    ):
        level = (phenotype_level or "multi_doc").strip().lower()
        if level not in {"single_doc", "multi_doc"}:
            level = "multi_doc"
        self.phenotype_level = level
        required_notes = {"patient_icn", "doc_id", "text"}
        if not required_notes.issubset(set(notes_df.columns)):
            raise ValueError(f"Notes missing {required_notes}")
        required_ann = {"round_id", "unit_id", "doc_id", "label_id", "reviewer_id", "label_value"}
        if not required_ann.issubset(set(ann_df.columns)):
            raise ValueError(f"Annotations missing {required_ann}")

        self.notes = notes_df.copy()
        self.notes["patient_icn"] = self.notes["patient_icn"].astype(str)
        self.notes["doc_id"] = self.notes["doc_id"].astype(str)
        self.notes["text"] = self.notes["text"].astype(str).map(normalize_text)
        original_unit_ids = None
        if "unit_id" in self.notes.columns:
            original_unit_ids = self.notes["unit_id"].astype(str)
        self._notes_by_doc_cache: Optional[Dict[str, str]] = None
        self._unit_doc_lookup: dict[str, str] = {}

        if "notetype" not in self.notes.columns:
            self.notes["notetype"] = ""
        else:
            self.notes["notetype"] = self.notes["notetype"].fillna("").astype(str)

        if self.phenotype_level == "single_doc":
            unit_ids = self.notes["doc_id"]
            if original_unit_ids is not None:
                doc_ids = self.notes["doc_id"].astype(str)
                for raw, doc in zip(original_unit_ids.tolist(), doc_ids.tolist()):
                    if raw:
                        self._unit_doc_lookup.setdefault(raw, doc)
        else:
            unit_ids = self.notes["patient_icn"]
        self.notes["unit_id"] = unit_ids.astype(str)

        self.ann = ann_df.copy()
        # --- Normalize multi-typed label_value columns ---
        # 1) String label_value (categorical/binary, free text)
        if "label_value" not in self.ann.columns:
            self.ann["label_value"] = ""
        self.ann["label_value"] = self.ann["label_value"].astype(str).str.strip()

        # 2) Numeric
        if "label_value_num" in self.ann.columns:
            self.ann["label_value_num"] = pd.to_numeric(self.ann["label_value_num"], errors="coerce")
        else:
            self.ann["label_value_num"] = np.nan

        # 3) Date
        if "label_value_date" in self.ann.columns:
            self.ann["label_value_date"] = pd.to_datetime(self.ann["label_value_date"], errors="coerce", utc=False)
        else:
            self.ann["label_value_date"] = pd.NaT

        for col in ("unit_id", "doc_id", "label_id", "reviewer_id"):
            self.ann[col] = self.ann[col].astype(str)
        if "labelset_id" in self.ann.columns:
            self.ann["labelset_id"] = self.ann["labelset_id"].astype(str)
        else:
            self.ann["labelset_id"] = ""
        if "document_text" in self.ann.columns:
            self.ann["document_text"] = self.ann["document_text"].astype(str).map(normalize_text)
        else:
            self.ann["document_text"] = ""
        for col in ("rationales_json", "document_metadata_json"):
            if col in self.ann.columns:
                self.ann[col] = self.ann[col].apply(safe_json_loads)
            else:
                self.ann[col] = None
        for col in ("label_rules", "reviewer_notes"):
            if col in self.ann.columns:
                self.ann[col] = self.ann[col].astype(str)
            else:
                self.ann[col] = ""

        self.label_rules_by_label = self._collect_label_rules()
        self._round_labelset_map = self._collect_round_labelsets()

    def unit_metadata(self) -> pd.DataFrame:
        columns = ["unit_id", "patient_icn"]
        if self.phenotype_level == "single_doc":
            columns.append("doc_id")
        meta = self.notes[columns].drop_duplicates(subset=["unit_id"]).copy()
        return meta

    def _collect_label_rules(self) -> Dict[str, str]:
        rules = {}
        if "label_rules" in self.ann.columns:
            df = self.ann[["label_id", "label_rules"]].dropna()
            for lid, grp in df.groupby("label_id"):
                vals = [v for v in grp["label_rules"].tolist() if isinstance(v, str) and v.strip()]
                if vals:
                    rules[lid] = vals[-1]
        return rules

    def _collect_round_labelsets(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        if {"round_id", "labelset_id"}.issubset(self.ann.columns):
            subset = self.ann[["round_id", "labelset_id"]].dropna()
            if not subset.empty:
                subset = subset.astype({"round_id": str, "labelset_id": str})
                for row in subset.drop_duplicates().itertuples(index=False):
                    if row.labelset_id:
                        mapping[str(row.round_id)] = str(row.labelset_id)
        return mapping

    def labelset_for_round(self, round_identifier: str) -> Optional[str]:
        return self._round_labelset_map.get(str(round_identifier))

    def exclude_units(self, unit_ids: set[str] | list[str]) -> int:
        """Remove excluded units from notes/annotations and reset caches.

        Returns the number of notes rows removed.
        """
        if not unit_ids:
            return 0

        excluded = {str(u) for u in unit_ids if str(u)}
        if self.notes.empty or not excluded:
            return 0

        mask = ~self.notes["unit_id"].astype(str).isin(excluded)
        removed_notes = int(len(self.notes) - mask.sum())
        if removed_notes:
            self.notes = self.notes.loc[mask].reset_index(drop=True)
            self._notes_by_doc_cache = None

        if not self.ann.empty and "unit_id" in self.ann.columns and excluded:
            ann_mask = ~self.ann["unit_id"].astype(str).isin(excluded)
            if int(len(self.ann) - ann_mask.sum()):
                self.ann = self.ann.loc[ann_mask].reset_index(drop=True)

        # Trim any stale unit->doc lookup entries for single_doc mode
        if self._unit_doc_lookup:
            for uid in list(self._unit_doc_lookup.keys()):
                if uid in excluded:
                    self._unit_doc_lookup.pop(uid, None)

        return removed_notes

    def labelset_for_annotation(
        self,
        unit_id: str,
        label_id: str,
        *,
        round_id: Optional[str] = None,
    ) -> Optional[str]:
        sub = self.ann
        if round_id is not None:
            sub = sub[sub["round_id"].astype(str) == str(round_id)]
        sub = sub[(sub["unit_id"].astype(str) == str(unit_id)) & (sub["label_id"].astype(str) == str(label_id))]
        if sub.empty:
            return None
        first = sub["labelset_id"].dropna().astype(str)
        for value in first:
            text = value.strip()
            if text:
                return text
        return None

    def notes_by_doc(self) -> Dict[str, str]:
        if self._notes_by_doc_cache is None:
            self._notes_by_doc_cache = dict(zip(self.notes["doc_id"].tolist(), self.notes["text"].tolist()))
        return self._notes_by_doc_cache

    def doc_id_for_unit(self, unit_id: str) -> Optional[str]:
        if self.phenotype_level != "single_doc":
            return None
        unit_str = str(unit_id)
        if not unit_str:
            return None
        notes = self.notes_by_doc()
        if unit_str in notes:
            return unit_str
        doc = self._unit_doc_lookup.get(unit_str)
        if doc:
            return doc
        matches = self.ann[self.ann["unit_id"] == unit_str]
        if not matches.empty:
            docs = matches["doc_id"].dropna().astype(str)
            if not docs.empty:
                return docs.iloc[0]
        return None

    def label_types(self) -> Dict[str, str]:
        """
        Infer label type per label_id using the three input columns:
          - If any non-null in label_value_num   -> "numeric"
          - elif any non-null in label_value_date -> "date"
          - else infer from string: binary vs categorical
        """
        types: Dict[str, str] = {}
        # group once to avoid repeated scans
        for lid, g in self.ann.groupby("label_id", sort=False):
            # numeric wins
            if g["label_value_num"].notna().any():
                types[str(lid)] = "numeric"
                continue
            # date next
            if g["label_value_date"].notna().any():
                types[str(lid)] = "date"
                continue
            # else string; decide binary vs categorical heuristically
            vals = g["label_value"].astype(str).str.lower().str.strip()
            uniq = set(v for v in vals if v not in ("", "nan", "none"))
            # common binary token set
            bin_tokens = {"0", "1", "true", "false", "present", "absent", "yes", "no", "neg", "pos", "positive", "negative", "unknown"}
            if uniq and uniq.issubset(bin_tokens):
                types[str(lid)] = "binary"
            else:
                types[str(lid)] = "categorical"  # use 'categorical' instead of 'text'
        return types

    def reviewer_disagreement(self, round_policy: str = 'last',
                          decay_half_life: float = 2.0) -> pd.DataFrame:
        """
        Compute a per-(unit_id,label_id) disagreement score in [0,1].
          - categorical/binary: normalized entropy
          - numeric: reviewer range / global IQR (clipped to [0,1])
          - date: span_days / global IQR_days (clipped)
        """
        ann = self.ann.copy()

        # Round selection
        try:
            ann["_round_ord"] = pd.to_numeric(ann["round_id"], errors="coerce")
            ord_series = ann["_round_ord"].fillna(ann["round_id"].astype("category").cat.codes)
        except Exception:
            ord_series = ann["round_id"].astype("category").cat.codes
        ann["_round_ord"] = ord_series
        last_ord = int(ann["_round_ord"].max()) if len(ann) else 0
        if round_policy == "last":
            ann = ann[ann["_round_ord"] == last_ord]
        # else 'all' or 'decay' -> keep all; decay applied later

        types = self.label_types()

        # Precompute global IQR per label for numeric/date
        iqr_num: Dict[str, float] = {}
        iqr_days: Dict[str, float] = {}

        for lid, g in ann.groupby("label_id", sort=False):
            t = types.get(str(lid), "categorical")
            if t == "numeric" and g["label_value_num"].notna().any():
                arr = g["label_value_num"].dropna().to_numpy(dtype="float64")
                if arr.size >= 2:
                    iqr = float(np.subtract(*np.percentile(arr, [75, 25])))
                    iqr_num[str(lid)] = iqr if iqr > 0 else 1.0
                else:
                    iqr_num[str(lid)] = 1.0
            elif t == "date" and g["label_value_date"].notna().any():
                dt = g["label_value_date"].dropna()
                if not dt.empty:
                    ords = dt.map(lambda x: x.toordinal()).to_numpy(dtype="int64")
                    iqr = float(np.subtract(*np.percentile(ords, [75, 25])))
                    iqr_days[str(lid)] = iqr if iqr > 0 else 1.0
                else:
                    iqr_days[str(lid)] = 1.0

        rows = []
        # group by unit, label, reviewer to assemble per (unit,label) reviewer values
        for (uid, lid), g in ann.groupby(["unit_id", "label_id"], sort=False):
            t = types.get(str(lid), "categorical")
            score = 0.0

            if t in ("categorical", "binary"):
                vals = g["label_value"].astype(str).str.lower().str.strip()
                uniq, cnts = np.unique([v for v in vals if v not in ("", "nan", "none")], return_counts=True)
                total = cnts.sum() if cnts.size else 0
                if total > 0:
                    p = cnts / total
                    ent = -np.sum(p * np.log2(np.clip(p, 1e-12, 1)))
                    ent_max = np.log2(max(len(uniq), 2))
                    score = float(ent / ent_max) if ent_max > 0 else 0.0
                else:
                    score = 0.0

            elif t == "numeric" and g["label_value_num"].notna().any():
                arr = g["label_value_num"].dropna().to_numpy(dtype="float64")
                if arr.size >= 2:
                    rng = float(np.nanmax(arr) - np.nanmin(arr))
                    denom = max(1e-6, iqr_num.get(str(lid), 1.0))
                    score = float(max(0.0, min(1.0, rng / (4.0 * denom))))  # 0..1
                else:
                    score = 0.0

            elif t == "date" and g["label_value_date"].notna().any():
                dt = g["label_value_date"].dropna()
                if len(dt) >= 2:
                    ords = dt.map(lambda x: x.toordinal()).to_numpy(dtype="int64")
                    span = float(np.nanmax(ords) - np.nanmin(ords))  # days
                    denom = max(1e-6, iqr_days.get(str(lid), 1.0))
                    score = float(max(0.0, min(1.0, span / (4.0 * denom))))  # 0..1
                else:
                    score = 0.0

            # optional decay weight: downweight older rounds relative to latest
            # Use half-life in rounds, e.g., half-life=2 means weight 0.5 two rounds ago
            if round_policy == "decay":
                decay_ord = g["_round_ord"].iloc[0]
                delta = float(last_ord - decay_ord)
                if delta > 0:
                    weight = 0.5 ** (delta / float(decay_half_life))
                    score *= weight

            labelset_value = None
            if "labelset_id" in g.columns:
                labelset_non_empty = g["labelset_id"].dropna().astype(str).str.strip()
                if not labelset_non_empty.empty:
                    labelset_value = labelset_non_empty.iloc[-1]

            round_value = None
            if "round_id" in g.columns:
                round_non_empty = g["round_id"].dropna().astype(str).str.strip()
                if not round_non_empty.empty:
                    round_value = str(round_non_empty.iloc[-1])

            rows.append({
                "unit_id": str(uid),
                "label_id": str(lid),
                "disagreement_score": float(score),
                "n_reviewers": int(g["reviewer_id"].nunique()),
                "round_id": round_value,
                "labelset_id": labelset_value,
            })
        return pd.DataFrame(rows)

    def hard_disagree(self, label_types: dict, *, date_days: int = 14, num_abs: float = 1.0, num_rel: float = 0.20) -> pd.DataFrame:
        """Return per-(unit_id,label_id) hard disagreement flags based on absolute/relative numeric and date-day spans.
        categorical/binary: hard=True if multiple unique non-empty values
        numeric: max pairwise |Δ| > max(num_abs, num_rel * max(|v_i|))
        date: span_days > date_days
        """
        rows = []
        for (uid, lid), g in self.ann.groupby(["unit_id", "label_id"], sort=False):
            t = label_types.get(str(lid), "categorical")
            hard = False
            if t in ("categorical", "binary"):
                vals = g["label_value"].astype(str).str.lower().str.strip()
                uniq = [v for v in vals.unique().tolist() if v not in ("", "nan", "none")]
                hard = (len(uniq) >= 2)
            elif t == "numeric" and g["label_value_num"].notna().any():
                arr = g["label_value_num"].dropna().to_numpy(dtype="float64")
                if arr.size >= 2:
                    vmax = float(np.nanmax(np.abs(arr)))
                    thresh = max(float(num_abs), float(num_rel) * max(1.0, vmax))
                    dif = float(np.nanmax(arr) - np.nanmin(arr))
                    hard = dif > thresh
            elif t == "date" and g["label_value_date"].notna().any():
                dt = g["label_value_date"].dropna()
                if len(dt) >= 2:
                    ords = dt.map(lambda x: x.toordinal()).to_numpy(dtype="int64")
                    span = float(np.nanmax(ords) - np.nanmin(ords))
                    hard = span > float(date_days)
            rows.append({"unit_id": str(uid), "label_id": str(lid), "hard_disagree": bool(hard)})
        return pd.DataFrame(rows)

    def get_prior_rationales(self, unit_id: str, label_id: str) -> List[dict]:
        sub = self.ann[(self.ann["unit_id"] == unit_id) & (self.ann["label_id"] == label_id)]
        spans = []
        for r in sub.itertuples(index=False):
            lst = r.rationales_json
            if isinstance(lst, list):
                for sp in lst:
                    if isinstance(sp, dict) and sp.get("snippet"):
                        spans.append(sp)
        return spans

    def last_round_consensus(self) -> Dict[Tuple[str, str], str]:
        """
        Returns {(unit_id,label_id): consensus_value_as_string} for the last round.
        Numeric/date are summarized by robust medians; categorical/binary by mode.
        """
        ann = self.ann.copy()
        # establish last round ordering
        try:
            ann["_round_ord"] = pd.to_numeric(ann["round_id"], errors="coerce")
            ord_series = ann["_round_ord"].fillna(ann["round_id"].astype("category").cat.codes)
        except Exception:
            ord_series = ann["round_id"].astype("category").cat.codes
        ann["_round_ord"] = ord_series
        last_ord = int(ann["_round_ord"].max()) if len(ann) else 0
        ann = ann[ann["_round_ord"] == last_ord]

        # Infer types on the last round only (cheaper)
        types = self.label_types()
        out: Dict[Tuple[str, str], str] = {}
        for (uid, lid), g in ann.groupby(["unit_id", "label_id"], sort=False):
            t = types.get(str(lid), "categorical")
            if t == "numeric" and g["label_value_num"].notna().any():
                med = np.nanmedian(g["label_value_num"].to_numpy(dtype="float64"))
                out[(str(uid), str(lid))] = str(med)
            elif t == "date" and g["label_value_date"].notna().any():
                # median date: convert to ordinal days, then back
                dt = g["label_value_date"].dropna()
                if not dt.empty:
                    ords = dt.map(lambda x: x.toordinal()).to_numpy(dtype="int64")
                    med_ord = int(np.median(ords))
                    out[(str(uid), str(lid))] = str(pd.Timestamp.fromordinal(med_ord).date())
                else:
                    out[(str(uid), str(lid))] = ""
            else:
                # categorical/binary: mode of string
                vals = g["label_value"].astype(str).str.lower().str.strip()
                if len(vals):
                    mode = vals.mode(dropna=True)
                    out[(str(uid), str(lid))] = str(mode.iloc[0]) if len(mode) else ""
                else:
                    out[(str(uid), str(lid))] = ""
        return out
