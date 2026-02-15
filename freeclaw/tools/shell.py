import os
import selectors
import subprocess
import time
from typing import Any

from .fs import ToolContext


_DEFAULT_BLOCKLIST = {
    # Prevent users from re-introducing shell parsing/metacharacters.
    "bash",
    "sh",
    "zsh",
    "fish",
    "cmd",
    "powershell",
    "pwsh",
    # Avoid privilege escalation / host changes.
    "sudo",
    "doas",
}

_NETWORK_TOOLS = {
    "curl",
    "wget",
    "ssh",
    "scp",
    "sftp",
    "nc",
    "ncat",
    "netcat",
    "telnet",
}


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    # Best-effort: don't leak obvious secrets into shell commands.
    for k in [
        "NVIDIA_API_KEY",
        "NIM_API_KEY",
        "NVIDIA_NIM_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "DISCORD_BOT_TOKEN",
        "FREECLAW_DISCORD_TOKEN",
    ]:
        env.pop(k, None)
    return env


def exec_argv(
    ctx: ToolContext,
    *,
    argv: list[str],
    stdin: str | None = None,
    env: dict[str, str] | None = None,
    timeout_s: float,
    max_output_bytes: int,
    block_network: bool,
    cwd: str | os.PathLike[str] | None = None,
    tool_name: str = "sh_exec",
) -> dict[str, Any]:
    if not isinstance(argv, list) or not argv:
        raise ValueError("argv must be a non-empty list of strings")
    argv = [str(x) for x in argv]
    if not argv[0].strip():
        raise ValueError("argv[0] is required")

    exe = os.path.basename(argv[0]).strip()
    if exe in _DEFAULT_BLOCKLIST:
        raise ValueError(f"Command blocked: {exe}")
    if block_network and exe in _NETWORK_TOOLS:
        raise ValueError(f"Network command blocked: {exe} (set FREECLAW_SHELL_BLOCK_NETWORK=false to allow)")

    to_s = float(timeout_s)
    if to_s <= 0:
        raise ValueError("timeout_s must be > 0")
    max_b = int(max_output_bytes)
    if max_b <= 0:
        raise ValueError("max_output_bytes must be > 0")

    start = time.time()
    child_env = _clean_env()
    if env:
        for k, v in env.items():
            child_env[str(k)] = str(v)

    proc = subprocess.Popen(
        argv,
        cwd=str(cwd or ctx.root),
        env=child_env,
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if stdin is not None and proc.stdin is not None:
        try:
            proc.stdin.write((stdin or "").encode("utf-8"))
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    sel = selectors.DefaultSelector()
    assert proc.stdout is not None
    assert proc.stderr is not None
    sel.register(proc.stdout, selectors.EVENT_READ, data="stdout")
    sel.register(proc.stderr, selectors.EVENT_READ, data="stderr")

    out_b = bytearray()
    err_b = bytearray()
    truncated = False
    timed_out = False

    def total_len() -> int:
        return len(out_b) + len(err_b)

    try:
        while True:
            if proc.poll() is not None and not sel.get_map():
                break

            if time.time() - start > to_s:
                timed_out = True
                proc.kill()
                break

            if total_len() > max_b:
                truncated = True
                proc.kill()
                break

            events = sel.select(timeout=0.05)
            if not events:
                if proc.poll() is not None:
                    pass
                continue

            for key, _mask in events:
                stream = key.fileobj
                kind = key.data
                try:
                    chunk = stream.read(8192)
                except Exception:
                    chunk = b""
                if not chunk:
                    try:
                        sel.unregister(stream)
                    except Exception:
                        pass
                    continue
                if kind == "stdout":
                    out_b.extend(chunk)
                else:
                    err_b.extend(chunk)
    finally:
        try:
            sel.close()
        except Exception:
            pass
        try:
            proc.stdout.close()
        except Exception:
            pass
        try:
            proc.stderr.close()
        except Exception:
            pass

    try:
        rc = proc.wait(timeout=0.2)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        rc = proc.poll()

    sig = None
    if isinstance(rc, int) and rc < 0:
        sig = int(-rc)

    if total_len() > max_b:
        truncated = True
        out_keep = min(len(out_b), max_b)
        err_keep = min(len(err_b), max(0, max_b - out_keep))
        out_b = out_b[:out_keep]
        err_b = err_b[:err_keep]

    return {
        "ok": True,
        "tool": str(tool_name),
        "cwd": str(cwd or ctx.root),
        "argv": argv,
        "exit_code": int(rc) if rc is not None else None,
        "signal": sig,
        "stdout": out_b.decode("utf-8", errors="replace"),
        "stderr": err_b.decode("utf-8", errors="replace"),
        "truncated": bool(truncated),
        "timed_out": bool(timed_out),
        "seconds": round(time.time() - start, 3),
        "max_output_bytes": int(max_b),
        "timeout_s": float(to_s),
    }


def sh_exec(
    ctx: ToolContext,
    *,
    argv: list[str],
    stdin: str | None = None,
    env: dict[str, str] | None = None,
    timeout_s: float | None = None,
    max_output_bytes: int | None = None,
) -> dict[str, Any]:
    if not ctx.shell_enabled:
        raise ValueError("sh_exec is disabled (set FREECLAW_ENABLE_SHELL=true or pass --enable-shell).")
    to_s = float(timeout_s) if timeout_s is not None else float(ctx.shell_timeout_s)
    max_b = int(max_output_bytes) if max_output_bytes is not None else int(ctx.shell_max_output_bytes)
    return exec_argv(
        ctx,
        argv=argv,
        stdin=stdin,
        env=env,
        timeout_s=to_s,
        max_output_bytes=max_b,
        block_network=bool(ctx.shell_block_network),
        cwd=ctx.root,
        tool_name="sh_exec",
    )
