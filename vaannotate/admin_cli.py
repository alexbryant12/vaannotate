"""Command-line admin tools."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import sqlite3
import typer
from rich import print
from rich.table import Table

from .corpus import bulk_import_from_csv
from .metrics import cohens_kappa, fleiss_kappa, percent_agreement
from .project import add_labelset, add_phenotype, build_project_paths, get_connection, init_project, register_reviewer
from .rounds import RoundBuilder
from .utils import ensure_dir

app = typer.Typer(help="VAAnnotate admin CLI")


@app.command()
def init(
    project_dir: Path = typer.Argument(..., help="Project directory"),
    project_id: str = typer.Option(..., help="Unique project identifier"),
    name: str = typer.Option(..., help="Project display name"),
    created_by: str = typer.Option("cli", help="Creator identifier"),
) -> None:
    """Initialize project folder."""
    paths = init_project(project_dir, project_id, name, created_by)
    print(f"Initialized project at {paths.root}")


@app.command()
def import_corpus(
    project_dir: Path = typer.Argument(..., help="Project directory (used as base for relative paths)"),
    patients_csv: Path = typer.Option(..., help="CSV file of patients"),
    documents_csv: Path = typer.Option(..., help="CSV file of documents"),
    corpus_db: Path = typer.Option(..., help="Destination SQLite database to create or overwrite"),
) -> None:
    base = project_dir.resolve()
    target = corpus_db if corpus_db.is_absolute() else (base / corpus_db)
    bulk_import_from_csv(target, patients_csv, documents_csv)
    print(f"Corpus imported to {target}")


@app.command()
def addreviewer(
    project_dir: Path = typer.Argument(...),
    reviewer_id: str = typer.Option(...),
    name: str = typer.Option(...),
    email: Optional[str] = typer.Option(None),
    windows_account: Optional[str] = typer.Option(None),
) -> None:
    with get_connection(build_project_paths(project_dir).project_db) as conn:
        register_reviewer(conn, reviewer_id, name, email, windows_account)
        conn.commit()
    print(f"Registered reviewer {reviewer_id}")


@app.command()
def addphenotype(
    project_dir: Path = typer.Argument(...),
    pheno_id: str = typer.Option(...),
    project_id: str = typer.Option(...),
    name: str = typer.Option(...),
    level: str = typer.Option(...),
    description: Optional[str] = typer.Option(None),
    corpus_path: Path = typer.Option(..., help="Path to the corpus SQLite database for this phenotype"),
) -> None:
    with get_connection(build_project_paths(project_dir).project_db) as conn:
        add_phenotype(
            conn,
            pheno_id,
            project_id,
            name,
            level,
            description,
            corpus_path=str(corpus_path),
            default_corpus_id=None,
        )
        conn.commit()
    print(f"Added phenotype {pheno_id}")


@app.command()
def createlabelset(
    project_dir: Path = typer.Argument(...),
    config_json: Path = typer.Option(..., help="JSON file describing label set"),
) -> None:
    data = json.loads(Path(config_json).read_text("utf-8"))
    with get_connection(build_project_paths(project_dir).project_db) as conn:
        project_row = conn.execute(
            "SELECT project_id FROM projects ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
        project_id = data.get("project_id") or (project_row["project_id"] if project_row else None)
        if not project_id:
            raise typer.BadParameter("Project metadata missing; ensure the project has been initialized")
        add_labelset(
            conn,
            labelset_id=data["labelset_id"],
            project_id=project_id,
            pheno_id=data.get("pheno_id"),
            version=data.get("version", 1),
            created_by=data.get("created_by", "cli"),
            notes=data.get("notes"),
            labels=data["labels"],
        )
        conn.commit()
    print(f"Label set {data['labelset_id']} stored")


@app.command()
def generateround(
    project_dir: Path = typer.Argument(...),
    pheno_id: str = typer.Option(...),
    config_json: Path = typer.Option(...),
    created_by: str = typer.Option("cli"),
) -> None:
    builder = RoundBuilder(project_dir)
    result = builder.generate_round(pheno_id, config_json, created_by)
    print(json.dumps(result, indent=2))


@app.command()
def importassignment(
    project_dir: Path = typer.Argument(...),
    pheno_id: str = typer.Option(...),
    round_number: int = typer.Option(...),
    reviewer_id: str = typer.Option(...),
) -> None:
    builder = RoundBuilder(project_dir)
    path = builder.import_assignment(pheno_id, round_number, reviewer_id)
    print(f"Imported assignment from {reviewer_id} -> {path}")


@app.command()
def aggregate(
    project_dir: Path = typer.Argument(...),
    pheno_id: str = typer.Option(...),
    round_number: int = typer.Option(...),
) -> None:
    builder = RoundBuilder(project_dir)
    path = builder.build_round_aggregate(pheno_id, round_number)
    print(f"Aggregate DB at {path}")


@app.command()
def iaa(
    project_dir: Path = typer.Argument(...),
    pheno_id: str = typer.Option(...),
    round_number: int = typer.Option(...),
    label_id: str = typer.Option(...),
) -> None:
    round_dir = Path(project_dir) / "phenotypes" / pheno_id / "rounds" / f"round_{round_number}"
    aggregate_db = round_dir / "round_aggregate.db"
    unit_map = defaultdict(dict)
    with sqlite3.connect(aggregate_db) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute(
            "SELECT unit_id, reviewer_id, value FROM unit_annotations WHERE label_id=?",
            (label_id,),
        ):
            unit_map[row["unit_id"]][row["reviewer_id"]] = row["value"]
    reviewers = sorted({reviewer for assignments in unit_map.values() for reviewer in assignments.keys()})
    if not reviewers:
        print("No annotations found for label")
        return
    if len(reviewers) == 2:
        pairs = []
        for assignments in unit_map.values():
            if reviewers[0] in assignments and reviewers[1] in assignments:
                pairs.append((assignments[reviewers[0]], assignments[reviewers[1]]))
        kappa = cohens_kappa(pairs)
        print(f"Cohen's kappa: {kappa:.3f}")
    else:
        categories = sorted({value for assignments in unit_map.values() for value in assignments.values() if value is not None})
        table = Table(title="Fleiss kappa input")
        table.add_column("Unit")
        for cat in categories:
            table.add_column(cat)
        matrix = []
        for unit_id, assignments in unit_map.items():
            counter = Counter(value for value in assignments.values() if value is not None)
            row_counts = [counter.get(cat, 0) for cat in categories]
            matrix.append(row_counts)
            table.add_row(unit_id, *[str(c) for c in row_counts])
        if matrix:
            print(table)
            print(f"Fleiss kappa: {fleiss_kappa(matrix):.3f}")
    unit_values = [list(assignments.values()) for assignments in unit_map.values()]
    print(f"Percent agreement: {percent_agreement(unit_values):.3f}")


@app.command()
def export_annotations(
    project_dir: Path = typer.Argument(...),
    pheno_id: str = typer.Option(...),
    round_number: int = typer.Option(...),
) -> None:
    round_dir = Path(project_dir) / "phenotypes" / pheno_id / "rounds" / f"round_{round_number}"
    aggregate_db = round_dir / "round_aggregate.db"
    exports_dir = ensure_dir(round_dir / "reports" / "exports")
    csv_path = exports_dir / "gold_annotations.csv"
    jsonl_path = exports_dir / "gold_annotations.jsonl"
    with sqlite3.connect(aggregate_db) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM unit_annotations").fetchall()
    default_fields = [
        "round_id",
        "unit_id",
        "reviewer_id",
        "label_id",
        "value",
        "value_num",
        "value_date",
        "na",
        "notes",
    ]
    fieldnames = list(rows[0].keys()) if rows else default_fields
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row)) + "\n")
    print(f"Exports written to {exports_dir}")


if __name__ == "__main__":
    app()
