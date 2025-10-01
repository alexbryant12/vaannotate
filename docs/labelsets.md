# Label set authoring

Label sets are versioned artefacts stored in `project.db`. Each label set
JSON payload contains the metadata needed to render Admin and Reviewer
UIs.

## JSON structure

```json
{
  "labelset_id": "ls_example_v1",
  "pheno_id": "ph_example",
  "version": 1,
  "created_by": "admin",
  "notes": "Initial release",
  "labels": [
    {
      "label_id": "has_condition",
      "name": "Has Condition",
      "type": "boolean",
      "order_index": 1,
      "required": true,
      "rules": "Mark Yes if the condition is present anywhere in the window.",
      "options": [
        {"option_id": "opt_yes", "value": "Yes", "display": "Yes"},
        {"option_id": "opt_no", "value": "No", "display": "No"},
        {"option_id": "opt_unknown", "value": "Unknown", "display": "Unknown"}
      ]
    },
    {
      "label_id": "severity",
      "name": "Severity",
      "type": "ordinal",
      "order_index": 2,
      "gating_expr": "has_condition == 'Yes'",
      "options": [
        {"option_id": "severity_none", "value": "None", "display": "None", "weight": 0},
        {"option_id": "severity_mild", "value": "Mild", "display": "Mild", "weight": 1},
        {"option_id": "severity_moderate", "value": "Moderate", "display": "Moderate", "weight": 2},
        {"option_id": "severity_severe", "value": "Severe", "display": "Severe", "weight": 3}
      ]
    }
  ]
}
```

## Field semantics

* `type` — Supported types are `boolean`, `categorical_single`,
  `categorical_multi`, `ordinal`, `integer`, `float`, `date`, and `text`.
* `required` — Boolean flag controlling gating in the Reviewer client.
* `gating_expr` — Lightweight expression language evaluated client-side
  to show or hide dependent fields.
* `na_allowed` — If `true`, the Reviewer client must allow an explicit
  NA toggle and audit the reason.
* `options` — Ordered list for discrete label types. Include `weight` for
  ordinal labels to support weighted κ.

Each edit to the label definitions should bump the `version` and use a
new `labelset_id`. Rounds always reference a specific version, preserving
provenance.
