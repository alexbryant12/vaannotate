"""Command line interface for VAAnnotate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .corpus import ensure_corpus, load_documents_csv, load_patients_csv
from .project import ProjectPaths, ensure_pheno, init_project, register_labelset
from .rounds import generate_round


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="VAAnnotate administrative toolkit")
    parser.add_argument("project_root", type=Path, help="Root directory for the project")
    parser.add_argument("command", choices=["init", "load-corpus", "create-phenotype", "register-labelset", "generate-round"], help="Command to execute")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Additional command arguments")
    parser.add_argument("--version", action="version", version=f"vaannotate {__version__}")

    parsed = parser.parse_args(argv)
    root: Path = parsed.project_root
    command: str = parsed.command
    rest: list[str] = parsed.args

    paths = ProjectPaths(root)

    if command == "init":
        init_parser = argparse.ArgumentParser(prog="vaannotate init")
        init_parser.add_argument("project_id")
        init_parser.add_argument("name")
        init_parser.add_argument("created_by")
        args = init_parser.parse_args(rest)
        init_project(root, args.project_id, args.name, args.created_by)
        ensure_corpus(paths)
    elif command == "load-corpus":
        corpus_parser = argparse.ArgumentParser(prog="vaannotate load-corpus")
        corpus_parser.add_argument("patients_csv", type=Path)
        corpus_parser.add_argument("documents_csv", type=Path)
        args = corpus_parser.parse_args(rest)
        ensure_corpus(paths)
        load_patients_csv(paths, args.patients_csv)
        load_documents_csv(paths, args.documents_csv)
    elif command == "create-phenotype":
        pheno_parser = argparse.ArgumentParser(prog="vaannotate create-phenotype")
        pheno_parser.add_argument("project_id")
        pheno_parser.add_argument("pheno_id")
        pheno_parser.add_argument("name")
        pheno_parser.add_argument("level", choices=["single_doc", "multi_doc"])
        pheno_parser.add_argument("description")
        args = pheno_parser.parse_args(rest)
        ensure_pheno(paths, args.pheno_id, args.project_id, args.name, args.level, args.description)
    elif command == "register-labelset":
        label_parser = argparse.ArgumentParser(prog="vaannotate register-labelset")
        label_parser.add_argument("labelset_json", type=Path)
        args = label_parser.parse_args(rest)
        data = json.loads(args.labelset_json.read_text("utf-8"))
        register_labelset(
            paths,
            labelset_id=data["labelset_id"],
            pheno_id=data["pheno_id"],
            version=data.get("version", 1),
            created_by=data.get("created_by", "admin"),
            notes=data.get("notes"),
            labels=data.get("labels", []),
        )
    elif command == "generate-round":
        round_parser = argparse.ArgumentParser(prog="vaannotate generate-round")
        round_parser.add_argument("pheno_id")
        round_parser.add_argument("round_number", type=int)
        round_parser.add_argument("config", type=Path)
        round_parser.add_argument("created_by")
        args = round_parser.parse_args(rest)
        generate_round(paths, config_path=args.config, pheno_id=args.pheno_id, round_number=args.round_number, created_by=args.created_by)
    else:
        parser.error(f"Unknown command {command}")


if __name__ == "__main__":  # pragma: no cover
    main()
