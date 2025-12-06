"""Helpers for label dependency graphs and gating rules."""
from __future__ import annotations

import math
import re
from typing import Dict, List, Tuple


def _parse_date(x):
    try:
        import pandas as _pd
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        dt = _pd.to_datetime(x, errors="coerce")
        if dt is _pd.NaT:
            return None
        return dt
    except Exception:
        return None


def _to_number(x):
    try:
        if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
            return float(x)
        s = str(x).strip()
        s = s.replace(',', '')
        return float(s)
    except Exception:
        return None


def _canon_str(x):
    if x is None:
        return ""
    s = str(x)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _canon_cat(x):
    s = _canon_str(x).lower()
    if s in {"y", "yes", "true", "1", "present", "positive", "pos"}:
        return "yes"
    if s in {"n", "no", "false", "0", "absent", "negative", "neg"}:
        return "no"
    return s


def build_label_dependencies(label_config: dict) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], List[str]]:
    """Return (parent->children, child->parents, roots) from label_config.

    Supports either:
      label_config[<lid>]['gated_by'] = <parent or [parents]>
      label_config[<lid>]['gating_rules'] = list of rule dicts with keys:
          - parent: parent label_id (optional if 'gated_by' provided)
          - type: 'categorical'|'numeric'|'date'
          - op: one of 'in','notin','==','!=','>','>=','<','<=','between','outside','exists','notnull'
          - values/value/low/high/inclusive (depending on op)
      label_config[<lid>]['gating_logic'] = 'AND'|'OR' (default AND) across provided rules
    """
    parent_to_children: Dict[str, List[str]] = {}
    child_to_parents: Dict[str, List[str]] = {}
    if not isinstance(label_config, dict):
        return {}, {}, []
    for lid, cfg in label_config.items():
        if str(lid) == "_meta":
            continue
        if not isinstance(cfg, dict):
            continue
        gb = cfg.get('gated_by')
        parents: List[str] = []
        if gb:
            if isinstance(gb, (list, tuple, set)):
                parents.extend([str(x) for x in gb])
            else:
                parents.append(str(gb))
        rules = cfg.get('gating_rules') or []
        for r in (rules if isinstance(rules, list) else [rules]):
            p = None
            if isinstance(r, dict):
                p = r.get('parent') or r.get('gated_by') or r.get('field')
            if p:
                p = str(p)
                if p not in parents:
                    parents.append(p)
        if parents:
            for p in parents:
                parent_to_children.setdefault(str(p), []).append(str(lid))
                child_to_parents.setdefault(str(lid), []).append(str(p))
    all_labels = {str(k) for k in label_config.keys() if str(k) != "_meta"}
    roots = [lid for lid in all_labels if lid not in child_to_parents]
    return parent_to_children, child_to_parents, roots


def _check_rule(parent_value, parent_type: str, rule: dict) -> bool:
    """Evaluate a single gating rule for a parent value of a given type."""
    if parent_type == 'numeric':
        v = _to_number(parent_value)
        if v is None:
            return False
        op = str(rule.get('op', 'in')).lower()
        if op == 'between':
            lo = _to_number(rule.get('low', None))
            hi = _to_number(rule.get('high', None))
            inc = bool(rule.get('inclusive', True))
            if lo is None or hi is None:
                return False
            return (lo <= v <= hi) if inc else (lo < v < hi)
        elif op in ('>', '>=', '<', '<=', '==', '!='):
            val = _to_number(rule.get('value', rule.get('values', [None])[0] if isinstance(rule.get('values'), list) else None))
            if val is None:
                return False
            if op == '>':
                return v > val
            if op == '>=':
                return v >= val
            if op == '<':
                return v < val
            if op == '<=':
                return v <= val
            if op == '==':
                return v == val
            if op == '!=':
                return v != val
            return False
        elif op in ('in', 'notin'):
            vals = rule.get('values', [])
            vals = [_to_number(x) for x in (vals if isinstance(vals, list) else [vals])]
            vals = [x for x in vals if x is not None]
            ok = v in vals
            return ok if op == 'in' else (not ok)
        elif op in ('exists', 'notnull'):
            return v is not None
        else:
            return False
    elif parent_type == 'date':
        d = _parse_date(parent_value)
        if d is None:
            return False
        op = str(rule.get('op', '>')).lower()
        if op == 'between':
            lo = _parse_date(rule.get('low'))
            hi = _parse_date(rule.get('high'))
            inc = bool(rule.get('inclusive', True))
            if lo is None or hi is None:
                return False
            return (lo <= d <= hi) if inc else (lo < d < hi)
        elif op in ('>', '>=', '<', '<=', '==', '!='):
            val = _parse_date(rule.get('value', rule.get('values', [None])[0] if isinstance(rule.get('values'), list) else None))
            if val is None:
                return False
            if op == '>':
                return d > val
            if op == '>=':
                return d >= val
            if op == '<':
                return d < val
            if op == '<=':
                return d <= val
            if op == '==':
                return d == val
            if op == '!=':
                return d != val
            return False
        elif op in ('exists', 'notnull'):
            return d is not None
        else:
            return False
    else:
        s = _canon_cat(parent_value)
        op = str(rule.get('op', 'in')).lower()
        if op in ('in', 'notin'):
            vals = rule.get('values', [])
            vals = [_canon_cat(x) for x in (vals if isinstance(vals, list) else [vals])]
            ok = s in vals
            return ok if op == 'in' else (not ok)
        elif op in ('==', '!='):
            val = _canon_cat(rule.get('value', rule.get('values', [None])[0] if isinstance(rule.get('values'), list) else None))
            if op == '==':
                return s == val
            else:
                return s != val
        elif op in ('exists', 'notnull'):
            return len(s) > 0
        else:
            return s in {'yes', 'present', 'true', '1'}


def _gating_for_child(child_id: str, label_config: dict) -> dict:
    cfg = (label_config or {}).get(child_id, {}) if isinstance(label_config, dict) else {}
    rules = cfg.get('gating_rules') or []
    if isinstance(rules, dict):
        rules = [rules]
    logic = str(cfg.get('gating_logic', 'AND')).upper()
    parents_declared = cfg.get('gated_by')
    parents: List[str] = []
    if parents_declared:
        parents = parents_declared if isinstance(parents_declared, list) else [parents_declared]
        if not rules:
            rules = [{'parent': p, 'type': 'categorical', 'op': 'in', 'values': ['yes', 'present', 'true', '1']} for p in parents]
    for r in rules:
        if isinstance(r, dict) and 'parent' not in r:
            if parents:
                r['parent'] = parents[0]
    return {'rules': rules, 'logic': logic}


def evaluate_gating(child_id: str, unit_id: str, parent_preds: dict, label_types: dict, label_config: dict) -> bool:
    """Return True if child is eligible given current parent predictions."""
    g = _gating_for_child(child_id, label_config)
    rules = g.get('rules') or []
    if not rules:
        return True
    logic = g.get('logic', 'AND').upper()
    outcomes = []
    for r in rules:
        if not isinstance(r, dict):
            continue
        p = str(r.get('parent'))
        p_type = label_types.get(p, 'categorical')
        val = parent_preds.get((unit_id, p))
        outcomes.append(_check_rule(val, p_type, r))
    if not outcomes:
        return True
    return all(outcomes) if logic != 'OR' else any(outcomes)


__all__ = [
    "build_label_dependencies",
    "evaluate_gating",
]
