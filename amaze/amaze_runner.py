import os
import runpy
import sys
from pathlib import Path

from amaze.instrumentation import install
from amaze import annotations as _annotations
from amaze.policy import Policy
from amaze.state import PolicyViolation, RuntimeState
from amaze.reporting import generate_html_report, open_report_if_possible


_ANNOTATION_MARKERS = ("amaze_tool", "amaze_llm", "amaze_agent", "from amaze.annotations", "from amaze import")


def _uses_annotations(script_path: str) -> bool:
    """Return True if the agent script imports or uses amaze annotation decorators."""
    try:
        source = Path(script_path).read_text(encoding="utf-8")
        return any(marker in source for marker in _ANNOTATION_MARKERS)
    except OSError:
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m amaze.amaze_runner <script.py> [policy.json]")
        sys.exit(1)

    script = sys.argv[1]
    policy_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(__file__), "..", "policy.json"
    )

    agent_name = Path(script).stem          # "agent.py" → "agent"
    policy = Policy.load(policy_path)
    runtime = RuntimeState(policy, agent_name=agent_name)
    os.environ["TRACE_ID"] = runtime.trace_id

    print("[aMaze] runner started", flush=True)
    print(f"[aMaze] script={script}", flush=True)
    print(f"[aMaze] trace_id={runtime.trace_id}", flush=True)

    if _uses_annotations(script):
        print("[aMaze] mode=annotations", flush=True)
        _annotations.set_runtime(runtime)
    else:
        print("[aMaze] mode=langchain", flush=True)
        install(runtime)

    policy_violation = None
    script_error = None
    try:
        runpy.run_path(script, run_name="__main__")
    except PolicyViolation as e:
        policy_violation = e
    except Exception as e:
        script_error = e

    # End-of-run graph completeness check
    graph_failures = runtime.validate_graph_complete()

    all_failures = (
        runtime.assertion_failures
        + graph_failures
        + ([str(policy_violation)] if policy_violation else [])
    )

    runtime.passed = (len(all_failures) == 0 and script_error is None)
    runtime.write()
    if runtime.audit_path is not None:
        html_path = generate_html_report(runtime.audit_path)
        print(f"[STATE] wrote html report: {html_path}", flush=True)
        open_report_if_possible(html_path)
    _print_report(runtime, all_failures, script_error)

    if not runtime.passed:
        sys.exit(1)

    return runtime


def _print_report(runtime, failures, script_error):
    # Aggregate totals across all completed turns plus any uncommitted state
    all_turns = runtime.turns
    total_llm = sum(t["llm_calls"] for t in all_turns) + runtime.llm_calls
    total_indirect = sum(t["indirect_llm_calls"] for t in all_turns) + runtime.indirect_llm_calls
    total_tool = sum(t["tool_calls"] for t in all_turns) + runtime.tool_calls
    total_tokens = sum(t["total_tokens"] for t in all_turns) + runtime.total_tokens
    tool_by_name: dict = {}
    for t in all_turns:
        for k, v in t["tool_calls_by_name"].items():
            tool_by_name[k] = tool_by_name.get(k, 0) + v
    for k, v in runtime.tool_calls_by_name.items():
        tool_by_name[k] = tool_by_name.get(k, 0) + v

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"aMazeTest Run Report  [trace: {runtime.trace_id[:8]}]")
    print(sep)
    print(f"LLM calls (direct):   {total_llm}")
    print(f"LLM calls (indirect): {total_indirect}")
    print(f"Tool calls:           {total_tool}")
    if tool_by_name:
        for name, count in tool_by_name.items():
            print(f"  {name}: {count}")
    print(f"Total tokens:         {total_tokens}")
    print(f"Call sequence:        {runtime.call_sequence}")
    if runtime.turns:
        print(f"\nPer-turn breakdown ({len(runtime.turns)} turn(s)):")
        for t in runtime.turns:
            tools = t["tool_calls_by_name"]
            tool_str = ", ".join(f"{k}:{v}" for k, v in tools.items()) if tools else "-"
            print(
                f"  Turn {t['turn']}: llm={t['llm_calls']} tool={t['tool_calls']} "
                f"({tool_str}) tokens={t['total_tokens']} seq={t['call_sequence']}"
            )
    if failures:
        print(f"\nFAILED ({len(failures)} issue(s)):")
        for f in failures:
            print(f"  - {f}")
    if script_error:
        print(f"\nScript error: {script_error}")
    status = "PASSED" if runtime.passed else "FAILED"
    print(f"\nResult: {status}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
