"""Run endpoints — single test run and suite run with SSE streaming."""
import asyncio
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from gui.database import get_conn
from gui.models import RunTestIn, RunSuiteIn
from gui.runner import run_test_async, update_test_run_db, compute_outcome

router = APIRouter(prefix="/api/runs", tags=["runs"])


# ── Single test run ────────────────────────────────────────────────────────────

@router.post("/test")
async def start_single_run(body: RunTestIn):
    """Start a single test run (non-streaming). Returns run_id immediately."""
    conn = get_conn()
    tc = conn.execute(
        "SELECT * FROM test_cases WHERE name=?", (body.test_case_name,)
    ).fetchone()
    if not tc:
        conn.close()
        raise HTTPException(404, "Test case not found")

    cur = conn.execute(
        "INSERT INTO test_runs (test_case_name,policy_name,agent_name,prompt,expected_pass) "
        "VALUES (?,?,?,?,?)",
        (tc["name"], tc["policy_name"], tc["agent_name"], tc["prompt"], tc["expected_pass"])
    )
    run_id = cur.lastrowid
    conn.commit()
    conn.close()

    asyncio.create_task(_run_single(run_id, tc["policy_name"], tc["agent_name"],
                                    tc["prompt"], tc["expected_pass"]))
    return {"run_id": run_id}


async def _run_single(run_id, policy_name, agent_name, prompt, expected_pass):
    conn = get_conn()
    agent_row = conn.execute("SELECT file_path FROM agents WHERE name=?", (agent_name,)).fetchone()
    conn.close()
    agent_file = agent_row["file_path"] if agent_row else agent_name

    passed, log, audit_json, audit_html = await run_test_async(
        policy_name, agent_file, prompt, run_id, update_test_run_db
    )
    outcome = compute_outcome(passed, bool(expected_pass))
    conn = get_conn()
    conn.execute("UPDATE test_runs SET outcome=? WHERE id=?", (outcome, run_id))
    conn.commit()
    conn.close()


@router.get("/test/{run_id}/stream")
async def stream_single_run(run_id: int):
    """SSE stream of live log lines for a running or completed test run."""
    conn = get_conn()
    row = conn.execute("SELECT * FROM test_runs WHERE id=?", (run_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Run not found")

    tc_name = row["test_case_name"]
    policy_name = row["policy_name"]
    agent_name = row["agent_name"]
    prompt = row["prompt"]
    expected_pass = row["expected_pass"]

    conn = get_conn()
    agent_row = conn.execute("SELECT file_path FROM agents WHERE name=?", (agent_name,)).fetchone()
    conn.close()
    agent_file = agent_row["file_path"] if agent_row else agent_name

    async def event_gen():
        lines_queue: asyncio.Queue = asyncio.Queue()

        async def on_line(line: str):
            await lines_queue.put(line)

        async def db_update(rid, passed, log, audit_json, audit_html):
            outcome = compute_outcome(passed, bool(expected_pass))
            conn = get_conn()
            conn.execute(
                "UPDATE test_runs SET finished_at=datetime('now'),outcome=?,log_output=?,"
                "audit_json_path=?,audit_html_path=? WHERE id=?",
                (outcome, log, audit_json, audit_html, rid)
            )
            conn.commit()
            conn.close()
            await lines_queue.put(None)  # sentinel

        task = asyncio.create_task(
            run_test_async(policy_name, agent_file, prompt, run_id, db_update, on_line)
        )

        while True:
            item = await lines_queue.get()
            if item is None:
                break
            yield f"data: {json.dumps({'line': item})}\n\n"

        # Send final outcome
        conn = get_conn()
        final = conn.execute("SELECT outcome FROM test_runs WHERE id=?", (run_id,)).fetchone()
        conn.close()
        outcome = final["outcome"] if final else "fail"
        yield f"data: {json.dumps({'done': True, 'outcome': outcome})}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.get("/test/{run_id}")
def get_single_run(run_id: int):
    conn = get_conn()
    row = conn.execute("SELECT * FROM test_runs WHERE id=?", (run_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Run not found")
    return dict(row)


# ── Suite run ─────────────────────────────────────────────────────────────────

@router.post("/suite")
async def start_suite_run(body: RunSuiteIn):
    """Start a suite run. Returns suite_run_id immediately; execution is sequential."""
    conn = get_conn()
    suite = conn.execute("SELECT * FROM suites WHERE name=?", (body.suite_name,)).fetchone()
    if not suite:
        conn.close()
        raise HTTPException(404, "Suite not found")

    cases = conn.execute(
        "SELECT tc.* FROM suite_cases sc JOIN test_cases tc ON sc.test_case_id=tc.id "
        "WHERE sc.suite_id=? AND sc.enabled=1 ORDER BY sc.position",
        (suite["id"],)
    ).fetchall()

    cur = conn.execute(
        "INSERT INTO suite_runs (suite_id,suite_name,total_count) VALUES (?,?,?)",
        (suite["id"], suite["name"], len(cases))
    )
    suite_run_id = cur.lastrowid
    conn.commit()
    conn.close()

    asyncio.create_task(_run_suite(suite_run_id, [dict(tc) for tc in cases]))
    return {"suite_run_id": suite_run_id}


async def _run_suite(suite_run_id: int, cases: list[dict]):
    counts = {"pass": 0, "fail": 0, "xfail": 0, "xpass": 0}

    for tc in cases:
        conn = get_conn()
        agent_row = conn.execute(
            "SELECT file_path FROM agents WHERE name=?", (tc["agent_name"],)
        ).fetchone()
        agent_file = agent_row["file_path"] if agent_row else tc["agent_name"]

        cur = conn.execute(
            "INSERT INTO test_runs (suite_run_id,test_case_name,policy_name,agent_name,"
            "prompt,expected_pass) VALUES (?,?,?,?,?,?)",
            (suite_run_id, tc["name"], tc["policy_name"], tc["agent_name"],
             tc["prompt"], tc["expected_pass"])
        )
        run_id = cur.lastrowid
        conn.commit()
        conn.close()

        passed, log, audit_json, audit_html = await run_test_async(
            tc["policy_name"], agent_file, tc["prompt"], run_id, update_test_run_db
        )
        outcome = compute_outcome(passed, bool(tc["expected_pass"]))
        conn = get_conn()
        conn.execute("UPDATE test_runs SET outcome=? WHERE id=?", (outcome, run_id))
        conn.commit()
        conn.close()
        counts[outcome] = counts.get(outcome, 0) + 1

    conn = get_conn()
    conn.execute(
        "UPDATE suite_runs SET finished_at=datetime('now'),status='done',"
        "pass_count=?,fail_count=?,xfail_count=?,xpass_count=? WHERE id=?",
        (counts["pass"], counts["fail"], counts["xfail"], counts.get("xpass", 0), suite_run_id)
    )
    conn.commit()
    conn.close()


@router.get("/suite/{suite_run_id}/stream")
async def stream_suite_run(suite_run_id: int):
    """SSE: streams per-test status updates as the suite runs sequentially."""
    conn = get_conn()
    suite_row = conn.execute("SELECT * FROM suite_runs WHERE id=?", (suite_run_id,)).fetchone()
    if not suite_row:
        conn.close()
        raise HTTPException(404)
    cases = conn.execute(
        "SELECT tc.* FROM suite_cases sc "
        "JOIN test_cases tc ON sc.test_case_id=tc.id "
        "JOIN suites s ON sc.suite_id=s.id "
        "WHERE s.id=? AND sc.enabled=1 ORDER BY sc.position",
        (suite_row["suite_id"],)
    ).fetchall()
    conn.close()

    async def event_gen():
        counts = {"pass": 0, "fail": 0, "xfail": 0, "xpass": 0}

        for tc in cases:
            tc = dict(tc)
            conn = get_conn()
            agent_row = conn.execute(
                "SELECT file_path FROM agents WHERE name=?", (tc["agent_name"],)
            ).fetchone()
            agent_file = agent_row["file_path"] if agent_row else tc["agent_name"]
            cur = conn.execute(
                "INSERT INTO test_runs (suite_run_id,test_case_name,policy_name,agent_name,"
                "prompt,expected_pass) VALUES (?,?,?,?,?,?)",
                (suite_run_id, tc["name"], tc["policy_name"], tc["agent_name"],
                 tc["prompt"], tc["expected_pass"])
            )
            run_id = cur.lastrowid
            conn.commit()
            conn.close()

            yield f"data: {json.dumps({'test': tc['name'], 'status': 'running', 'run_id': run_id})}\n\n"

            lines_queue: asyncio.Queue = asyncio.Queue()
            done_event = asyncio.Event()

            async def on_line(line, q=lines_queue):
                await q.put(line)

            async def db_upd(rid, passed, log, audit_json, audit_html,
                             exp=tc["expected_pass"], q=lines_queue, ev=done_event):
                outcome = compute_outcome(passed, bool(exp))
                c = get_conn()
                c.execute(
                    "UPDATE test_runs SET finished_at=datetime('now'),outcome=?,log_output=?,"
                    "audit_json_path=?,audit_html_path=? WHERE id=?",
                    (outcome, log, audit_json, audit_html, rid)
                )
                c.commit()
                c.close()
                await q.put(None)
                ev.set()

            task = asyncio.create_task(
                run_test_async(tc["policy_name"], agent_file, tc["prompt"],
                               run_id, db_upd, on_line)
            )

            while not done_event.is_set() or not lines_queue.empty():
                try:
                    item = await asyncio.wait_for(lines_queue.get(), timeout=0.1)
                    if item is None:
                        break
                    yield f"data: {json.dumps({'test': tc['name'], 'log': item})}\n\n"
                except asyncio.TimeoutError:
                    continue

            await task
            passed_val, _, _, _ = task.result()
            outcome = compute_outcome(passed_val, bool(tc["expected_pass"]))
            counts[outcome] = counts.get(outcome, 0) + 1

            conn = get_conn()
            conn.execute("UPDATE test_runs SET outcome=? WHERE id=?", (outcome, run_id))
            conn.commit()
            conn.close()

            yield f"data: {json.dumps({'test': tc['name'], 'status': 'done', 'outcome': outcome, 'run_id': run_id})}\n\n"

        # Final summary
        conn = get_conn()
        conn.execute(
            "UPDATE suite_runs SET finished_at=datetime('now'),status='done',"
            "pass_count=?,fail_count=?,xfail_count=?,xpass_count=? WHERE id=?",
            (counts["pass"], counts["fail"], counts["xfail"], counts.get("xpass", 0), suite_run_id)
        )
        conn.commit()
        conn.close()
        yield f"data: {json.dumps({'done': True, 'counts': counts})}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.get("/suite/{suite_run_id}")
def get_suite_run(suite_run_id: int):
    conn = get_conn()
    row = conn.execute("SELECT * FROM suite_runs WHERE id=?", (suite_run_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404)
    result = dict(row)
    test_runs = conn.execute(
        "SELECT * FROM test_runs WHERE suite_run_id=? ORDER BY id",
        (suite_run_id,)
    ).fetchall()
    result["test_runs"] = [dict(r) for r in test_runs]
    conn.close()
    return result


@router.get("/suite-history/{suite_name}")
def suite_history(suite_name: str):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM suite_runs WHERE suite_name=? ORDER BY id DESC LIMIT 20",
        (suite_name,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
