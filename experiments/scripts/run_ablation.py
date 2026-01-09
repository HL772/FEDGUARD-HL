import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_once(root: Path, label: str, config_path: str, output_dir: Path, extra_env: dict) -> None:
    env = os.environ.copy()
    env.update(extra_env)
    env["CONFIG_PATH"] = config_path
    cmd = [sys.executable, "experiments/scripts/run_local_demo.py"]
    print(f"[ablation] running {label} config={config_path}", flush=True)
    subprocess.run(cmd, cwd=root, env=env, check=True)

    metrics_src = root / "server" / "metrics" / "metrics.jsonl"
    if metrics_src.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_dst = output_dir / f"{label}_metrics.jsonl"
        shutil.copyfile(metrics_src, metrics_dst)
        config_dst = output_dir / f"{label}_config.yaml"
        shutil.copyfile(root / config_path, config_dst)
        print(f"[ablation] saved {metrics_dst}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run baseline vs innovation ablation.")
    parser.add_argument("--baseline-config", default="experiments/configs/baseline.yaml")
    parser.add_argument("--innovation-config", default="experiments/configs/innovation.yaml")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--innovation-label", default="innovation")
    parser.add_argument("--output-dir", default="experiments/results")
    args = parser.parse_args()

    root = _project_root()
    output_dir = root / args.output_dir

    shared_env = {
        "MAX_ROUNDS": os.environ.get("MAX_ROUNDS", "10"),
        "NUM_CLIENTS": os.environ.get("NUM_CLIENTS", "6"),
        "CLIENTS_PER_ROUND": os.environ.get("CLIENTS_PER_ROUND", "6"),
        "CLIENT_BATCH_SIZE": os.environ.get("CLIENT_BATCH_SIZE", "2"),
        "ONLINE_TTL_SEC": os.environ.get("ONLINE_TTL_SEC", "60"),
    }

    _run_once(root, args.baseline_label, args.baseline_config, output_dir, shared_env)
    _run_once(root, args.innovation_label, args.innovation_config, output_dir, shared_env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
