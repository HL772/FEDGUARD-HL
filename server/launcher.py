import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional


# 本地多进程启动器（用于 Dashboard 手动启动训练）
# 只负责启动/停止客户端进程，不再启动新的 Server 进程。


def ensure_mnist_ready(root: Path, download: bool) -> None:
    processed = root / "data" / "MNIST" / "processed" / "training.pt"
    if processed.exists():
        return
    if not download:
        return
    cmd = [
        sys.executable,
        "-u",
        "-c",
        (
            "from client.data.partition import load_mnist;"
            "load_mnist('data', download=True);"
            "print('[launcher] MNIST download complete', flush=True)"
        ),
    ]
    print("[launcher] preparing MNIST dataset", flush=True)
    subprocess.run(cmd, cwd=root, check=True)


class LocalDemoRunner:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._client_procs: List[subprocess.Popen] = []
        self.last_error: str = ""
        self.last_status: str = "idle"
        self.last_params: Dict[str, object] = {}

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> str:
        if self.is_running():
            return "running"
        return self.last_status or "idle"

    def start(
        self,
        *,
        num_clients: int,
        client_batch_size: int,
        max_rounds: int,
        config_path: str,
        server_url: Optional[str] = None,
        download_data: bool = False,
        stay_online_on_not_selected: bool = False,
        request_timeout: Optional[float] = None,
        join_timeout: Optional[float] = None,
        heartbeat_interval: Optional[float] = None,
    ) -> bool:
        with self._lock:
            if self.is_running():
                return False
            self.last_error = ""
            self.last_status = "running"
            self.last_params = {
                "num_clients": num_clients,
                "client_batch_size": client_batch_size,
                "max_rounds": max_rounds,
                "config_path": config_path,
                "server_url": server_url,
                "download_data": download_data,
                "stay_online_on_not_selected": stay_online_on_not_selected,
            }
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                kwargs={
                    "num_clients": num_clients,
                    "client_batch_size": client_batch_size,
                    "max_rounds": max_rounds,
                    "config_path": config_path,
                    "server_url": server_url,
                    "download_data": download_data,
                    "stay_online_on_not_selected": stay_online_on_not_selected,
                    "request_timeout": request_timeout,
                    "join_timeout": join_timeout,
                    "heartbeat_interval": heartbeat_interval,
                },
                daemon=True,
            )
            self._thread.start()
            return True

    def stop(self) -> None:
        self._stop_event.set()
        self._terminate_all()
        self.last_status = "stopped"

    def _spawn(self, cmd: List[str], name: str) -> subprocess.Popen:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # 防止本地回环请求走系统代理导致 502
        env.setdefault("NO_PROXY", "*")
        env.setdefault("no_proxy", "*")
        proc = subprocess.Popen(cmd, cwd=self._root, env=env)
        print(f"[launcher] started {name} pid={proc.pid}", flush=True)
        self._client_procs.append(proc)
        return proc

    def _spawn_heartbeat(
        self,
        *,
        client_name: str,
        extra_args: List[str],
        config_path: str,
        download_data: bool,
    ) -> subprocess.Popen:
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "client.main",
            "--client-name",
            client_name,
            "--heartbeat-only",
            "--config",
            config_path,
        ]
        if extra_args:
            cmd.extend(extra_args)
        if download_data:
            cmd.append("--download-data")
        return self._spawn(cmd, f"{client_name}-heartbeat")

    def _terminate_all(self) -> None:
        for proc in list(self._client_procs):
            if proc.poll() is not None:
                continue
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._client_procs = [proc for proc in self._client_procs if proc.poll() is None]

    def _run(
        self,
        *,
        num_clients: int,
        client_batch_size: int,
        max_rounds: int,
        config_path: str,
        server_url: Optional[str],
        download_data: bool,
        stay_online_on_not_selected: bool,
        request_timeout: Optional[float],
        join_timeout: Optional[float],
        heartbeat_interval: Optional[float],
    ) -> None:
        try:
            ensure_mnist_ready(self._root, download=download_data)
            extra_args: List[str] = []
            if request_timeout is not None:
                extra_args.extend(["--request-timeout", str(request_timeout)])
            if join_timeout is not None:
                extra_args.extend(["--join-timeout", str(join_timeout)])
            if heartbeat_interval is not None:
                extra_args.extend(["--heartbeat-interval", str(heartbeat_interval)])
            if stay_online_on_not_selected:
                extra_args.append("--stay-online-on-not-selected")
            if server_url:
                extra_args.extend(["--server-url", server_url])
                print(f"[launcher] server_url={server_url}", flush=True)

            batch_size = max(1, min(int(client_batch_size), int(num_clients)))

            # 先注册所有客户端
            for idx in range(1, num_clients + 1):
                if self._stop_event.is_set():
                    return
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
                if extra_args:
                    cmd.extend(extra_args)
                if download_data:
                    cmd.append("--download-data")
                proc = self._spawn(cmd, f"{name}-register")
                proc.wait()

            if stay_online_on_not_selected:
                # 启动常驻心跳，避免批次训练导致客户端全部离线
                for idx in range(1, num_clients + 1):
                    if self._stop_event.is_set():
                        return
                    name = f"client-{idx}"
                    self._spawn_heartbeat(
                        client_name=name,
                        extra_args=extra_args,
                        config_path=config_path,
                        download_data=download_data,
                    )

            # 按轮次启动客户端（批次并发）
            for round_id in range(1, max_rounds + 1):
                if self._stop_event.is_set():
                    return
                print(f"[launcher] starting round {round_id}", flush=True)
                for start in range(0, num_clients, batch_size):
                    if self._stop_event.is_set():
                        return
                    procs: List[subprocess.Popen] = []
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
                        if extra_args:
                            cmd.extend(extra_args)
                        if download_data:
                            cmd.append("--download-data")
                        procs.append(self._spawn(cmd, name))
                    for proc in procs:
                        proc.wait()
        except Exception as exc:
            self.last_error = str(exc)
            self.last_status = "error"
            print(f"[launcher] failed: {exc}", flush=True)
        finally:
            self._terminate_all()
            if self.last_status != "error":
                if self._stop_event.is_set():
                    self.last_status = "stopped"
                else:
                    self.last_status = "completed"
