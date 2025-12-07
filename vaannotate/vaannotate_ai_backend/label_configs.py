from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Tuple


def sanitize_label_config(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Return a shallow copy of ``config`` without metadata entries."""

    if not isinstance(config, Mapping):
        return {}
    return {key: value for key, value in dict(config).items() if str(key) != "_meta"}


@dataclass(frozen=True)
class LabelConfigBundle:
    """Collection of label configurations for the AI backend.

    Attributes
    ----------
    current:
        The label configuration for the round currently being built.
    current_labelset_id:
        Identifier of the current round's label set.
    legacy:
        Mapping of legacy ``labelset_id`` values to their materialised label
        configurations.  The current label set may also be present here for
        convenience when code paths expect legacy lookups by identifier.
    round_labelsets:
        Mapping of known ``round_id`` values (and/or round numbers encoded as
        strings) to their associated ``labelset_id``.
    """

    current: Optional[Dict[str, object]] = None
    current_labelset_id: Optional[str] = None
    legacy: Dict[str, Dict[str, object]] = field(default_factory=dict)
    round_labelsets: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.current is not None:
            object.__setattr__(self, "current", sanitize_label_config(self.current))
        sanitized_legacy = {
            str(labelset_id): sanitize_label_config(config)
            for labelset_id, config in (self.legacy or {}).items()
        }
        object.__setattr__(self, "legacy", sanitized_legacy)

    def with_current_fallback(self, label_config: Mapping[str, object] | None) -> "LabelConfigBundle":
        """Return a bundle where ``current`` is populated from ``label_config``.

        This is primarily used to maintain compatibility with existing call
        sites that still supply a single ``label_config`` dictionary.
        """

        if label_config is None:
            return self
        sanitized = sanitize_label_config(label_config)
        if self.current == sanitized:
            return self
        return replace(self, current=sanitized)

    def config_for_labelset(self, labelset_id: Optional[str]) -> Dict[str, object]:
        """Return the materialised label configuration for ``labelset_id``."""

        if not labelset_id:
            return self.current or {}
        normalized = str(labelset_id)
        if self.current_labelset_id and normalized == str(self.current_labelset_id):
            if self.current is not None:
                return self.current
        return self.legacy.get(normalized, self.current or {})

    def config_for_round(self, round_identifier: Optional[str]) -> Dict[str, object]:
        """Return the configuration associated with ``round_identifier``."""

        if not round_identifier:
            return self.current or {}
        labelset_id = self.round_labelsets.get(str(round_identifier))
        return self.config_for_labelset(labelset_id)

    @staticmethod
    def _normalize_type(raw: Optional[object]) -> Optional[str]:
        if raw is None:
            return None
        text = str(raw).strip().lower()
        if not text:
            return None
        mapping = {
            "boolean": "binary",
            "bool": "binary",
            "yes/no": "binary",
            "yesno": "binary",
            "y/n": "binary",
            "yn": "binary",
            "binary": "binary",
            "categorical": "categorical",
            "categorical_single": "categorical_single",
            "categorical_multi": "categorical_multi",
            "category": "categorical",
            "multiclass": "categorical",
            "multi": "categorical",
            "multi_select": "categorical_multi",
            "options": "categorical",
            "option": "categorical",
            "text": "categorical",
            "string": "categorical",
            "free_text": "categorical",
            "numeric": "numeric",
            "number": "numeric",
            "int": "numeric",
            "integer": "numeric",
            "float": "numeric",
            "double": "numeric",
            "decimal": "numeric",
            "ordinal": "ordinal",
            "rank": "ordinal",
            "ranking": "ordinal",
            "date": "date",
            "datetime": "date",
            "timestamp": "date",
        }
        return mapping.get(text, "categorical")

    @staticmethod
    def _extract_rule_text(entry: Mapping[str, object] | None) -> Optional[str]:
        if not isinstance(entry, Mapping):
            return None
        for key in ("rule", "rules", "why", "query", "text"):
            val = entry.get(key)
            if isinstance(val, str):
                text = val.strip()
                if text:
                    return text
            elif isinstance(val, list):
                for item in reversed(val):
                    if isinstance(item, str):
                        text = item.strip()
                        if text:
                            return text
                    elif isinstance(item, Mapping):
                        text = str(item.get("text") or item.get("rule") or "").strip()
                        if text:
                            return text
            elif isinstance(val, Mapping):
                text = str(val.get("text") or val.get("rule") or "").strip()
                if text:
                    return text
        return None

    def legacy_rules_map(self) -> dict[str, str]:
        legacy_rules: Dict[str, str] = {}
        for config in (self.legacy or {}).values():
            for key, entry in (config or {}).items():
                if str(key) == "_meta":
                    continue
                label_entry = entry if isinstance(entry, Mapping) else {}
                raw_id = label_entry.get("label_id") if isinstance(label_entry, Mapping) else None
                label_id = str(raw_id or key).strip()
                if not label_id:
                    continue

                rule_text = self._extract_rule_text(label_entry)
                if rule_text is not None:
                    legacy_rules[label_id] = rule_text
                elif label_id not in legacy_rules:
                    legacy_rules[label_id] = ""
        return legacy_rules

    def legacy_label_types(self) -> dict[str, str]:
        legacy_types: Dict[str, str] = {}
        for config in (self.legacy or {}).values():
            for key, entry in (config or {}).items():
                if str(key) == "_meta":
                    continue
                label_entry = entry if isinstance(entry, Mapping) else {}
                raw_id = label_entry.get("label_id") if isinstance(label_entry, Mapping) else None
                label_id = str(raw_id or key).strip()
                if not label_id:
                    continue

                normalized_type = self._normalize_type(
                    label_entry.get("type") if isinstance(label_entry, Mapping) else None
                )
                if normalized_type:
                    legacy_types[label_id] = normalized_type
                elif label_id not in legacy_types:
                    legacy_types[label_id] = "categorical"
        return legacy_types

    def current_rules_map(self, label_config: Mapping[str, object] | None = None) -> dict[str, str]:
        if not (label_config or self.current):
            return {}

        config = label_config if label_config is not None else self.current or {}
        rules_map: Dict[str, str] = {}

        for key, entry in (config or {}).items():
            if str(key) == "_meta":
                continue
            label_entry = entry if isinstance(entry, Mapping) else {}
            raw_id = label_entry.get("label_id") if isinstance(label_entry, Mapping) else None
            label_id = str(raw_id or key).strip()
            if not label_id:
                continue

            rule_text = self._extract_rule_text(label_entry)
            if rule_text is not None:
                rules_map[label_id] = rule_text
            elif label_id not in rules_map:
                rules_map[label_id] = ""

        return rules_map

    def current_label_types(self, label_config: Mapping[str, object] | None = None) -> dict[str, str]:
        if not (label_config or self.current):
            return {}

        config = label_config if label_config is not None else self.current or {}
        label_types: Dict[str, str] = {}

        for key, entry in (config or {}).items():
            if str(key) == "_meta":
                continue
            label_entry = entry if isinstance(entry, Mapping) else {}
            raw_id = label_entry.get("label_id") if isinstance(label_entry, Mapping) else None
            label_id = str(raw_id or key).strip()
            if not label_id:
                continue

            normalized_type = self._normalize_type(
                label_entry.get("type") if isinstance(label_entry, Mapping) else None
            )
            if normalized_type:
                label_types[label_id] = normalized_type
            elif label_id not in label_types:
                label_types[label_id] = "categorical"

        return label_types


    def label_maps(
        self,
        label_config: Mapping[str, object] | None = None,
        ann_df: Any | None = None,
    ) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]:
        """Return (legacy_rules, legacy_types, current_rules, current_types).

        This centralizes the logic for:
        - deriving legacy rules/types from the bundle's ``legacy`` configs
        - optionally inferring a synthetic legacy config from annotations
        - deriving current rules/types from ``current`` (or an override)
        - falling back to legacy maps when current maps are empty
        """

        bundle = self

        # Start from any explicit legacy configs.
        legacy_rules_map = bundle.legacy_rules_map()
        legacy_label_types = bundle.legacy_label_types()

        # Optional annotation-based fallback when there is no legacy config.
        if not legacy_rules_map and ann_df is not None and hasattr(ann_df, "__iter__"):
            legacy_config: Dict[str, Dict[str, object]] = {}

            try:
                label_ids = {str(lid) for lid in ann_df.get("label_id", []) if str(lid)}
            except Exception:  # noqa: BLE001
                label_ids = set()

            for label_id in sorted(label_ids):
                rule_text: Optional[str] = None
                try:
                    # Mirror the existing ActiveLearningPipeline._label_maps logic:
                    # look at label_rules for this label and take the last non-empty string.
                    rules_col = ann_df.loc[ann_df["label_id"] == label_id, "label_rules"]
                    to_list = getattr(rules_col, "tolist", None)
                    if callable(to_list):
                        for raw_rule in reversed(to_list()):
                            if isinstance(raw_rule, str):
                                text = raw_rule.strip()
                                if text:
                                    rule_text = text
                                    break
                except Exception:  # noqa: BLE001
                    rule_text = None

                entry: Dict[str, object] = {"label_id": label_id, "type": "boolean"}
                if rule_text is not None:
                    entry["rules"] = rule_text
                legacy_config[label_id] = entry

            if legacy_config:
                # Materialise a synthetic legacy labelset based on annotations.
                fallback_bundle = replace(bundle, legacy={"_annotations": legacy_config})
                legacy_rules_map = fallback_bundle.legacy_rules_map()
                legacy_label_types = fallback_bundle.legacy_label_types()

        # Current rules/types, optionally overridden by a specific label_config.
        try:
            current_rules_map = bundle.current_rules_map(label_config)
        except TypeError:
            # Backwards-compatibility if the signature is used positionally.
            current_rules_map = bundle.current_rules_map()

        try:
            current_label_types = bundle.current_label_types(label_config)
        except TypeError:
            current_label_types = bundle.current_label_types()

        # If there is no explicit current config, fall back to legacy maps.
        if not current_rules_map and legacy_rules_map:
            current_rules_map = dict(legacy_rules_map)
        if not current_label_types and legacy_label_types:
            current_label_types = dict(legacy_label_types)

        return legacy_rules_map, legacy_label_types, current_rules_map, current_label_types


EMPTY_BUNDLE = LabelConfigBundle()
