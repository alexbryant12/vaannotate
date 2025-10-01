"""Console-based annotator client stub."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
import typer
from rich.prompt import Prompt

app = typer.Typer(help="Lightweight console annotator")


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


@app.command()
def open_assignment(assignment_dir: Path = typer.Argument(...)) -> None:
    """Interactive loop for annotating units."""
    db_path = assignment_dir / "assignment.db"
    if not db_path.exists():
        raise typer.BadParameter("assignment.db missing")
    with _connect(db_path) as conn:
        label_rows = conn.execute("SELECT DISTINCT label_id FROM annotations").fetchall()
        label_ids = [row["label_id"] for row in label_rows]
        typer.echo(f"Labels: {', '.join(label_ids)}")
        units = conn.execute("SELECT unit_id, complete FROM units ORDER BY display_rank").fetchall()
        for unit in units:
            typer.echo(f"\n=== Unit {unit['unit_id']} ===")
            values = {}
            for label_id in label_ids:
                existing = conn.execute(
                    "SELECT value, na FROM annotations WHERE unit_id=? AND label_id=?",
                    (unit["unit_id"], label_id),
                ).fetchone()
                prompt = f"{label_id} value" + (" (NA=leave blank)" if existing["na"] else "")
                response = Prompt.ask(prompt, default=existing["value"] or "")
                if response.strip():
                    conn.execute(
                        "UPDATE annotations SET value=?, na=0 WHERE unit_id=? AND label_id=?",
                        (response.strip(), unit["unit_id"], label_id),
                    )
                else:
                    conn.execute(
                        "UPDATE annotations SET value=NULL, na=1 WHERE unit_id=? AND label_id=?",
                        (unit["unit_id"], label_id),
                    )
            conn.execute(
                "UPDATE units SET complete=1, completed_at=? WHERE unit_id=?",
                (datetime.utcnow().isoformat(), unit["unit_id"]),
            )
            conn.execute(
                "INSERT INTO events(event_id, ts, actor, event_type, payload_json) VALUES (?,?,?,?,?)",
                (
                    f"evt_{unit['unit_id']}_{datetime.utcnow().timestamp()}",
                    datetime.utcnow().isoformat(),
                    "annotator",
                    "unit_completed",
                    json.dumps({"unit_id": unit["unit_id"]}),
                ),
            )
        conn.commit()
    typer.echo("Assignment complete.")


if __name__ == "__main__":
    app()
