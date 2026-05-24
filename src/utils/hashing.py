import hashlib
import json
from pathlib import Path


def file_hash(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()[:16]}"


def load_state(output_dir: Path) -> dict:
    state_file = output_dir / "state.json"
    if state_file.exists():
        return json.loads(state_file.read_text())
    return {}


def save_state(output_dir: Path, state: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "state.json").write_text(json.dumps(state, indent=2))


def source_id_from_filename(filename: str) -> str:
    h = hashlib.sha256(filename.encode()).hexdigest()[:8]
    stem = Path(filename).stem[:20].replace(" ", "_")
    return f"src_{stem}_{h}"
