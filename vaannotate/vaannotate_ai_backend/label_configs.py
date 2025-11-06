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


EMPTY_BUNDLE = LabelConfigBundle()
