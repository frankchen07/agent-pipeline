import json
from pathlib import Path
from typing import Iterator


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def update_jsonl_field(path: Path, key_field: str, key_value: str, updates: dict) -> None:
    records = read_jsonl(path)
    for rec in records:
        if rec.get(key_field) == key_value:
            rec.update(updates)
    write_jsonl(path, records)
