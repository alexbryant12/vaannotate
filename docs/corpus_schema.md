# Corpus schema

The corpus is stored in `corpus/corpus.db`. The SQLite database is
immutable after load and includes the following tables and indexes.

## patients

| column        | type | description |
| ------------- | ---- | ----------- |
| patient_icn   | TEXT PRIMARY KEY | Global patient identifier |
| sta3n         | TEXT | Facility identifier |
| date_index    | DATE | Optional anchor date for windowing |
| softlabel     | REAL | Optional pre-annotation score |

## documents

| column      | type | description |
| ----------- | ---- | ----------- |
| doc_id      | TEXT PRIMARY KEY | Unique document identifier |
| patient_icn | TEXT | Foreign key to `patients.patient_icn` |
| notetype    | TEXT | Canonicalized note type (e.g., `tiustandardtitle:PRIMARY CARE NOTE`) |
| note_year   | INTEGER | Calendar year of the note |
| date_note   | DATE | Full ISO date |
| cptname     | TEXT | Optional CPT-derived subtype |
| sta3n       | TEXT | Facility identifier |
| hash        | TEXT | SHA256 hash of canonicalized note text |
| text        | TEXT | Canonicalized note text |

## Indexes

* `idx_documents_patient` on `(patient_icn)`
* `idx_documents_notetype` on `(notetype)`
* `idx_documents_year` on `(note_year)`
* `idx_documents_sta` on `(sta3n)`

The ingestion helpers (`vaannotate load-corpus`) automatically
canonicalize newline and whitespace, enforce NFC normalization, and
persist the SHA256 hash. The hash is referenced later to guarantee
immutable corpora: if a stored document hash changes between sampling
and adjudication, the Admin workflow will abort.
