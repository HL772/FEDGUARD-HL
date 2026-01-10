import os
import subprocess
import sys
import time
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_mnist_ready(root: Path) -> None:
    processed = root / "data" / "MNIST" / "processed" / "training.pt"
    if processed.exists():
        print("[demo] MNIST already prepared", flush=True)
        return
    cmd = [
        sys.executable,
        "-u",
        "-c",
        (
            "from client.data.partition import load_mnist;"
            "load_mnist('data', download=True);"
            "print('[demo] MNIST download complete', flush=True)"
        ),
    ]
    print("[demo] preparing MNIST dataset", flush=True)
    subprocess.run(cmd, cwd=root, check=True)


def _load_config(path: str) -> dict:
    if not path:
        return {}
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
            return data or {}
    except Exception:
        return {}


def _spawn_process(cmd, cwd, name, extra_env=None):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)
    proc = subprocess.Popen(cmd, cwd=cwd, env=env) # 真实进程启动
    print(f"[demo] started {name} pid={proc.pid}", flush=True)
    return proc


def _terminate(proc, name):
    if proc is None or proc.poll() is not None:
        return
    print(f"[demo] stopping {name} pid={proc.pid}", flush=True)
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def main() -> int:
    root = _project_root()
    server_cmd = [sys.executable, "-u", "-m", "server.app"]
    server_proc = None
    client_procs = []

    try:
        _ensure_mnist_ready(root)
        config_path = os.environ.get("CONFIG_PATH", "experiments/configs/default.yaml")
        config = _load_config(config_path)
        train_cfg = config.get("train", {}) if isinstance(config, dict) else {}
        num_clients = int(os.environ.get("NUM_CLIENTS", "3"))
        clients_per_round = os.environ.get("CLIENTS_PER_ROUND") or str(num_clients)
        max_rounds = int(os.environ.get("MAX_ROUNDS", "3"))
        lr_env = os.environ.get("LR")
        epochs_env = os.environ.get("EPOCHS")
        lr_value = lr_env if lr_env is not None else str(train_cfg.get("lr", "0.03"))
        epochs_value = epochs_env if epochs_env is not None else str(
            train_cfg.get("epochs", "1")
        )
        server_env = {
            "MAX_ROUNDS": str(max_rounds),
            "CLIENTS_PER_ROUND": clients_per_round,
            "LR": lr_value,
            "BATCH_SIZE": os.environ.get("BATCH_SIZE", "32"),
            "EPOCHS": epochs_value,
            "CONFIG_PATH": config_path,
        }
        server_proc = _spawn_process(server_cmd, root, "server", extra_env=server_env)
        time.sleep(1.5)

        download_data = os.environ.get("DOWNLOAD_DATA", "0") == "1"
        client_extra_args = []
        request_timeout = os.environ.get("CLIENT_REQUEST_TIMEOUT")
        if request_timeout:
            client_extra_args.extend(["--request-timeout", request_timeout])
        join_timeout = os.environ.get("CLIENT_JOIN_TIMEOUT")
        if join_timeout:
            client_extra_args.extend(["--join-timeout", join_timeout])
        heartbeat_interval = os.environ.get("CLIENT_HEARTBEAT_INTERVAL")
        if heartbeat_interval:
            client_extra_args.extend(["--heartbeat-interval", heartbeat_interval])
        stay_online = os.environ.get("CLIENT_STAY_ONLINE_NOT_SELECTED")
        if stay_online == "1":
            client_extra_args.append("--stay-online-on-not-selected")
        batch_size = int(os.environ.get("CLIENT_BATCH_SIZE", "1"))
        if batch_size <= 0:
            batch_size = 1
        if batch_size > num_clients:
            batch_size = num_clients
        max_rounds = int(server_env["MAX_ROUNDS"])

        for idx in range(1, num_clients + 1):
            name = f"client-{idx}"
            cmd = [sys.executable, "-u", "-m", "client.main", "--client-name", name]
            cmd.extend(
                [
                    "--client-rank",
                    str(idx - 1),
                    "--num-clients",
                    str(num_clients),
                    "--register-only",
                    "--config",
                    config_path,
                ]
            )
            if client_extra_args:
                cmd.extend(client_extra_args)
            if download_data:
                cmd.append("--download-data")
            proc = _spawn_process(cmd, root, f"{name}-register")
            proc.wait()

        for round_id in range(1, max_rounds + 1):
            print(f"[demo] starting round {round_id}", flush=True)
            for start in range(0, num_clients, batch_size):
                client_procs = []
                for idx in range(start, min(start + batch_size, num_clients)):
                    name = f"client-{idx + 1}"
                    cmd = [sys.executable, "-u", "-m", "client.main", "--client-name", name]
                    cmd.extend(
                        [
                            "--client-rank",
                            str(idx),
                            "--num-clients",
                            str(num_clients),
                            "--single-round",
                            "--config",
                            config_path,
                        ]
                    )
                    if client_extra_args:
                        cmd.extend(client_extra_args)
                    if download_data:
                        cmd.append("--download-data")
                    client_procs.append(_spawn_process(cmd, root, name))
                for proc in client_procs:
                    proc.wait()
        print("[demo] all clients completed", flush=True)
        return 0
    except KeyboardInterrupt:
        print("[demo] interrupted", flush=True)
        return 1
    finally:
        for proc in client_procs:
            _terminate(proc, f"client-{proc.pid}")
        _terminate(server_proc, "server")


if __name__ == "__main__":
    raise SystemExit(main())
