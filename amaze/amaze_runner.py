import os
import runpy
import sys
from pathlib import Path

from amaze.instrumentation import install
from amaze.policy import Policy
from amaze.state import PolicyViolation, RuntimeState


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

    _print_report(runtime, all_failures, script_error)

    if not runtime.passed:
        sys.exit(1)

    return runtime


def _print_report(runtime, failures, script_error):
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"aMazeTest Run Report  [trace: {runtime.trace_id[:8]}]")
    print(sep)
    print(f"LLM calls (direct):   {runtime.llm_calls}")
    print(f"LLM calls (indirect): {runtime.indirect_llm_calls}")
    print(f"Tool calls:           {runtime.tool_calls}")
    if runtime.tool_calls_by_name:
        for name, count in runtime.tool_calls_by_name.items():
            print(f"  {name}: {count}")
    print(f"Total tokens:         {runtime.total_tokens}")
    print(f"Call sequence:        {runtime.call_sequence}")
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
