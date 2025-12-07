from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Optional


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
            "category": "categorical",
            "multiclass": "categorical",
            "multi": "categorical",
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


EMPTY_BUNDLE = LabelConfigBundle()
