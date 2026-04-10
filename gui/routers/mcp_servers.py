from fastapi import APIRouter, HTTPException
from gui.database import get_conn
from gui.models import McpServerIn

router = APIRouter(prefix="/api/mcp-servers", tags=["mcp-servers"])


@router.get("")
def list_mcp_servers():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM mcp_servers ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.post("", status_code=201)
def create_mcp_server(body: McpServerIn):
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO mcp_servers (name,url,transport,notes,env_json) VALUES (?,?,?,?,?)",
            (body.name, body.url, body.transport, body.notes, body.env_json)
        )
        conn.commit()
    except Exception as e:
        raise HTTPException(400, str(e))
    finally:
        conn.close()
    return {"ok": True}


@router.put("/{name}")
def update_mcp_server(name: str, body: McpServerIn):
    conn = get_conn()
    conn.execute(
        "UPDATE mcp_servers SET url=?,transport=?,notes=?,env_json=? WHERE name=?",
        (body.url, body.transport, body.notes, body.env_json, name)
    )
    conn.commit()
    conn.close()
    return {"ok": True}


@router.delete("/{name}")
def delete_mcp_server(name: str):
    conn = get_conn()
    conn.execute("DELETE FROM mcp_servers WHERE name=?", (name,))
    conn.commit()
    conn.close()
    return {"ok": True}


@router.post("/{name}/fetch-tools")
async def fetch_tools(name: str):
    """Attempt to connect to MCP server and call tools/list.
    Falls back to empty list on error; updates status and tools_json in DB."""
    import json, httpx
    conn = get_conn()
    row = conn.execute("SELECT * FROM mcp_servers WHERE name=?", (name,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "Server not found")
    url = row["url"]
    conn.close()

    tools = []
    status = "error"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}

            # Step 1: initialize to obtain session ID
            init_resp = await client.post(
                url,
                json={"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "amazetest-gui", "version": "0.1"},
                }},
                headers=headers,
            )
            session_id = init_resp.headers.get("mcp-session-id")
            if session_id:
                headers["mcp-session-id"] = session_id

            # Step 2: list tools
            resp = await client.post(
                url,
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
                headers=headers,
            )

            # Parse SSE or plain JSON
            data = None
            ct = resp.headers.get("content-type", "")
            if "text/event-stream" in ct:
                for line in resp.text.splitlines():
                    if line.startswith("data:"):
                        data = json.loads(line[len("data:"):].strip())
                        break
            else:
                data = resp.json()

            if data:
                tools = data.get("result", {}).get("tools", [])
            status = "ok"
    except Exception:
        status = "error"

    conn = get_conn()
    conn.execute(
        "UPDATE mcp_servers SET status=?, tools_json=? WHERE name=?",
        (status, json.dumps(tools), name)
    )
    conn.commit()
    conn.close()
    return {"status": status, "tools": tools}
