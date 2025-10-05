from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .database import Database

MetadataScope = str
MetadataType = str


@dataclass(frozen=True)
class MetadataField:
    key: str
    label: str
    scope: MetadataScope
    data_type: MetadataType
    expression: str
    alias: str
    constant_for_unit: bool


@dataclass
class MetadataFilterCondition:
    field: str
    label: str
    scope: MetadataScope
    data_type: MetadataType
    min_value: Optional[str] = None
    max_value: Optional[str] = None
    values: Optional[List[str]] = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "field": self.field,
            "label": self.label,
            "scope": self.scope,
            "type": self.data_type,
        }
        if self.min_value is not None:
            payload["min"] = self.min_value
        if self.max_value is not None:
            payload["max"] = self.max_value
        if self.values:
            payload["values"] = list(self.values)
        return payload

    @classmethod
    def from_payload(cls, data: Mapping[str, Any]) -> "MetadataFilterCondition":
        field = str(data.get("field") or "")
        label = str(data.get("label") or field)
        scope = str(data.get("scope") or "document")
        data_type = str(data.get("type") or "text")
        min_value = data.get("min")
        max_value = data.get("max")
        raw_values = data.get("values")
        values: Optional[List[str]]
        if isinstance(raw_values, (list, tuple, set)):
            values = [str(value) for value in raw_values]
        else:
            values = None
        return cls(
            field=field,
            label=label,
            scope=scope,
            data_type=data_type,
            min_value=str(min_value) if min_value is not None else None,
            max_value=str(max_value) if max_value is not None else None,
            values=values,
        )


_METADATA_EXCLUDE_KEYS = {
    "doc_id",
    "hash",
    "text",
    "order_index",
    "documents",
    "metadata",
    "metadata_json",
}


def _is_meaningful(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


def extract_document_metadata(source: Mapping[str, object] | None) -> Dict[str, object]:
    """Return a normalized metadata dictionary for a document payload.

    The helper merges explicit ``metadata`` dictionaries with any scalar
    attributes present on the document payload while skipping core fields like
    the note text and document identifier. Empty strings and ``None`` values
    are omitted so the caller receives only meaningful metadata entries.
    """

    metadata: Dict[str, object] = {}
    if not source:
        return metadata
    raw_metadata = source.get("metadata") if isinstance(source, Mapping) else None
    if isinstance(raw_metadata, Mapping):
        for key, value in raw_metadata.items():
            if _is_meaningful(value):
                metadata[key] = value
    for key, value in source.items():
        if key in _METADATA_EXCLUDE_KEYS:
            continue
        if isinstance(value, Mapping):
            continue
        if isinstance(value, (list, tuple, set)):
            continue
        if not _is_meaningful(value):
            continue
        metadata.setdefault(key, value)
    return metadata


def _sanitize_alias(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", value.strip())
    if not cleaned:
        cleaned = "field"
    if cleaned[0].isdigit():
        cleaned = f"f_{cleaned}"
    return cleaned


def _human_label(name: str) -> str:
    name = name.replace("_", " ").strip()
    if not name:
        return "Metadata"
    return name[:1].upper() + name[1:]


_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%m/%d/%Y",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%m-%d-%Y",
    "%m-%d-%Y %H:%M",
    "%m-%d-%Y %H:%M:%S",
    "%Y%m%d",
)


def _normalize_date_string(value: str) -> Optional[str]:
    value = value.strip()
    if not value:
        return None
    candidates = [value]
    if value.endswith("Z"):
        candidates.append(value[:-1] + "+00:00")
    for candidate in candidates:
        try:
            dt = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        return dt.date().isoformat()
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(value, fmt)
        except ValueError:
            continue
        return dt.date().isoformat()
    if re.match(r"^\d{4}-\d{1,2}-\d{1,2}T\d{2}:\d{2}", value):
        try:
            dt = datetime.fromisoformat(value[:16])
        except ValueError:
            return None
        return dt.date().isoformat()
    return None


def normalize_date_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = _normalize_date_string(str(value))
    return normalized or (str(value).strip() or None)


def _looks_like_date(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return _normalize_date_string(value) is not None


def _all_numeric(values: Sequence[Any]) -> bool:
    numbers_found = False
    for value in values:
        if value in (None, ""):
            continue
        try:
            float(value)
        except (TypeError, ValueError):
            return False
        numbers_found = True
    return numbers_found


def _all_dates(values: Sequence[Any]) -> bool:
    dates_found = False
    for value in values:
        if value in (None, ""):
            continue
        if not _looks_like_date(value):
            return False
        dates_found = True
    return dates_found


def _infer_data_type(declared_type: str, samples: Sequence[Any]) -> MetadataType:
    declared = (declared_type or "").upper()
    if any(token in declared for token in ("INT", "REAL", "NUM", "DEC", "DOUBLE", "FLOAT")):
        return "number"
    if "BOOL" in declared:
        return "number"
    if "DATE" in declared or "TIME" in declared:
        return "date"
    if _all_numeric(samples):
        return "number"
    if _all_dates(samples):
        return "date"
    return "text"


def _column_samples(conn: sqlite3.Connection, table: str, column: str, limit: int) -> List[Any]:
    sql = f'SELECT "{column}" FROM {table} WHERE "{column}" IS NOT NULL LIMIT ?'
    rows = conn.execute(sql, (limit,)).fetchall()
    return [row[0] for row in rows]


def _json_samples(conn: sqlite3.Connection, limit: int) -> Dict[str, List[Any]]:
    samples: Dict[str, List[Any]] = {}
    try:
        rows = conn.execute(
            "SELECT metadata_json FROM documents WHERE metadata_json IS NOT NULL LIMIT ?",
            (limit,),
        ).fetchall()
    except sqlite3.DatabaseError:
        return samples
    for row in rows:
        payload = row[0]
        if not payload:
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        for key, value in parsed.items():
            if isinstance(value, (list, dict)):
                continue
            samples.setdefault(key, []).append(value)
    return samples


def _is_constant_per_patient(conn: sqlite3.Connection, expression: str) -> bool:
    try:
        row = conn.execute(
            f"""
            SELECT 1
            FROM (
                SELECT patient_icn,
                       MIN({expression}) AS min_value,
                       MAX({expression}) AS max_value
                FROM documents
                GROUP BY patient_icn
                HAVING (min_value IS NOT max_value)
                LIMIT 1
            )
            """
        ).fetchone()
    except sqlite3.DatabaseError:
        return False
    return row is None


def _supports_json(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("SELECT json_extract('{\"a\": 1}', '$.a')").fetchone()
    except sqlite3.DatabaseError:
        return False
    return True


def _discover_from_connection(conn: sqlite3.Connection, sample_limit: int) -> List[MetadataField]:
    fields: List[MetadataField] = []

    patient_info = conn.execute("PRAGMA table_info(patients)").fetchall()
    for row in patient_info:
        name = row["name"]
        if name == "patient_icn":
            continue
        declared = row["type"]
        samples = _column_samples(conn, "patients", name, sample_limit)
        data_type = _infer_data_type(declared, samples)
        key = f"patient.{name}"
        alias = _sanitize_alias(f"patient__{name}")
        label = f"{_human_label(name)} (patient)"
        expression = f'patients."{name}"'
        fields.append(
            MetadataField(
                key=key,
                label=label,
                scope="patient",
                data_type=data_type,
                expression=expression,
                alias=alias,
                constant_for_unit=True,
            )
        )

    document_info = conn.execute("PRAGMA table_info(documents)").fetchall()
    excluded = {"doc_id", "patient_icn", "hash", "text"}
    for row in document_info:
        name = row["name"]
        if name in excluded:
            continue
        declared = row["type"]
        samples = _column_samples(conn, "documents", name, sample_limit)
        data_type = _infer_data_type(declared, samples)
        key = f"document.{name}"
        alias = _sanitize_alias(f"document__{name}")
        label = _human_label(name)
        expression = f'documents."{name}"'
        constant = _is_constant_per_patient(conn, expression)
        fields.append(
            MetadataField(
                key=key,
                label=label,
                scope="document",
                data_type=data_type,
                expression=expression,
                alias=alias,
                constant_for_unit=constant,
            )
        )

    if _supports_json(conn):
        json_samples = _json_samples(conn, sample_limit)
        for name, values in json_samples.items():
            if not values:
                continue
            data_type = _infer_data_type("", values)
            key = f"metadata.{name}"
            alias = _sanitize_alias(f"metadata__{name}")
            label = _human_label(name)
            escaped = name.replace('"', '""')
            expression = f"json_extract(documents.metadata_json, '$.\"{escaped}\"')"
            constant = _is_constant_per_patient(conn, expression)
            fields.append(
                MetadataField(
                    key=key,
                    label=label,
                    scope="document",
                    data_type=data_type,
                    expression=expression,
                    alias=alias,
                    constant_for_unit=constant,
                )
            )

    fields.sort(key=lambda field: (field.label.lower(), field.key))
    return fields


def discover_corpus_metadata(
    source: Database | sqlite3.Connection,
    *,
    sample_limit: int = 500,
) -> List[MetadataField]:
    if isinstance(source, Database):
        with source.connect() as conn:
            return _discover_from_connection(conn, sample_limit)
    return _discover_from_connection(source, sample_limit)
