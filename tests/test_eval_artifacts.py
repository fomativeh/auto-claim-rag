import json
import os
import subprocess
import sys
from pathlib import Path


def test_eval_writes_artifacts(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "runs"
    index_dir = tmp_path / "index"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")

    cmd = [
        sys.executable,
        "-m",
        "auto_claim_rag",
        "eval",
        "--llm",
        "null",
        "--out-dir",
        str(out_dir),
        "--index-dir",
        str(index_dir),
        "--claims-dir",
        str(project_root / "data" / "raw" / "claims"),
        "--kb-dir",
        str(project_root / "data" / "knowledge_base"),
        "--gold-path",
        str(project_root / "data" / "gold" / "claims_gold.jsonl"),
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(project_root),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    summary = json.loads(proc.stdout)
    run_dir = Path(summary["run_dir"])

    assert (run_dir / "predictions.jsonl").exists()
    assert (run_dir / "scores.jsonl").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "error_analysis.json").exists()
