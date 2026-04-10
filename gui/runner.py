"""Subprocess runner — executes amaze_runner.py and streams output via SSE."""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
RUNNER_MODULE = "amaze.amaze_runner"


async def run_test_async(
    policy_name: str,
    agent_file: str,
    prompt: str,
    test_run_id: int,
    db_update_fn,          # coroutine: (run_id, outcome, log, audit_json, audit_html) -> None
    line_callback=None,    # coroutine: (line: str) -> None  — for SSE streaming
):
    """Execute a single test in a subprocess, stream output, update DB when done."""
    from gui.database import get_conn

    policies_dir = ROOT / "policies"
    policy_file = policies_dir / f"{policy_name}.json"

    # Write policy JSON from DB if the file doesn't exist on disk
    if not policy_file.exists():
        conn = get_conn()
        row = conn.execute("SELECT policy_json FROM policies WHERE name=?", (policy_name,)).fetchone()
        conn.close()
        if row:
            policy_file.write_text(row["policy_json"], encoding="utf-8")

    agent_path = Path(agent_file)
    if not agent_path.is_absolute():
        agent_path = ROOT / agent_file

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    env["AGENT_PROMPT"] = prompt

    cmd = [
        PYTHON, "-m", RUNNER_MODULE,
        str(agent_path),
        str(policy_file),
    ]

    log_lines: list[str] = []
    audit_json_path = None
    audit_html_path = None
    passed = None

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=str(ROOT),
        )

        async for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip()
            log_lines.append(line)
            if line_callback:
                await line_callback(line)

            if "writing audit file:" in line:
                parts = line.split("writing audit file:")
                if len(parts) > 1:
                    audit_json_path = parts[1].strip()
            if "wrote html report:" in line:
                parts = line.split("wrote html report:")
                if len(parts) > 1:
                    audit_html_path = parts[1].strip()
            if line.startswith("RESULT:"):
                passed = "PASS" in line

        await proc.wait()
        if passed is None:
            passed = proc.returncode == 0

    except Exception as exc:
        line = f"[runner error] {exc}"
        log_lines.append(line)
        if line_callback:
            await line_callback(line)
        passed = False

    log_text = "\n".join(log_lines)
    await db_update_fn(test_run_id, passed, log_text, audit_json_path, audit_html_path)
    return passed, log_text, audit_json_path, audit_html_path


async def update_test_run_db(run_id: int, passed: bool, log: str, audit_json: str, audit_html: str):
    from gui.database import get_conn
    outcome_raw = "pass" if passed else "fail"
    conn = get_conn()
    conn.execute(
        "UPDATE test_runs SET finished_at=datetime('now'), outcome=?, log_output=?, "
        "audit_json_path=?, audit_html_path=? WHERE id=?",
        (outcome_raw, log, audit_json, audit_html, run_id)
    )
    conn.commit()
    conn.close()


def compute_outcome(passed: bool, expected_pass: bool) -> str:
    """Map (passed, expected_pass) → outcome label."""
    if expected_pass:
        return "pass" if passed else "fail"
    else:
        return "xfail" if not passed else "xpass"
