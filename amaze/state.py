import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from amaze.policy import (
    AssertionOperator,
    ControlPlanePolicy,
    GraphPolicy,
    MockConfig,
)


class PolicyViolation(Exception):
    pass


def _evaluate_assertion(operator: AssertionOperator, expected: Any, actual: Any) -> bool:
    s = str(actual)
    if operator == AssertionOperator.EQUALS:
        return actual == expected
    elif operator == AssertionOperator.CONTAINS:
        return str(expected) in s
    elif operator == AssertionOperator.STARTS_WITH:
        return s.startswith(str(expected))
    elif operator == AssertionOperator.MATCHES_REGEX:
        return bool(re.search(expected, s))
    return False


class RuntimeState:
    def __init__(self, policy, agent_name: str = "agent"):
        self.policy = policy
        self.agent_name = agent_name
        self.trace_id = str(uuid.uuid4())

        # Call counters
        self.llm_calls = 0          # direct LLM calls only
        self.indirect_llm_calls = 0
        self.tool_calls = 0
        self.tool_calls_by_name: dict = {}

        # Token tracking (bug fix: initialize here)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0

        # Graph mode state
        if isinstance(policy, GraphPolicy):
            self.current_node: str = policy.nodes[0]
            self._adjacency: dict = policy.adjacency()
        else:
            self.current_node = ""
            self._adjacency = {}

        self.call_sequence: list = []  # current turn's call nodes
        self.last_turn: dict = {}      # snapshot of completed turn stats (populated after each finish)

        # Context for indirect LLM hint injection
        self.last_llm_mock = None          # MockConfig used by the last direct LLM call
        self.last_tool_description: str = ""  # description of the last tool invoked

        self.events: list = []
        self.assertion_failures: list = []
        self.passed = False

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self, event_type: str, payload: dict):
        event = {
            "ts": time.time(),
            "trace_id": self.trace_id,
            "type": event_type,
            "payload": payload,
        }
        self.events.append(event)
        print(f"[STATE] {event_type} trace_id={self.trace_id} payload={payload}", flush=True)

    # ------------------------------------------------------------------
    # LLM entry
    # ------------------------------------------------------------------

    def enter_llm(self, model: str = "unknown", is_indirect: bool = False):
        if is_indirect:
            self.indirect_llm_calls += 1
            self.log("llm_call_indirect", {"count": self.indirect_llm_calls, "model": model})
            return

        # Control plane: enforce call limit
        if isinstance(self.policy, ControlPlanePolicy):
            if (self.policy.max_llm_calls is not None
                    and self.llm_calls >= self.policy.max_llm_calls):
                raise PolicyViolation(
                    f"LLM call limit {self.policy.max_llm_calls} exceeded"
                )

        # Graph mode: per-call validation
        if isinstance(self.policy, GraphPolicy):
            self.check_graph_step("llm")

        self.llm_calls += 1
        self.log("llm_call", {"count": self.llm_calls, "model": model})

    # ------------------------------------------------------------------
    # Tool entry
    # ------------------------------------------------------------------

    def enter_tool(self, tool_name: str, input_data):
        if isinstance(self.policy, ControlPlanePolicy):
            if self.policy.allowed_tools and tool_name not in self.policy.allowed_tools:
                raise PolicyViolation(f"Tool '{tool_name}' is not in the allowed list")

            if (self.policy.max_tool_calls is not None
                    and self.tool_calls >= self.policy.max_tool_calls):
                raise PolicyViolation(
                    f"Tool call limit {self.policy.max_tool_calls} exceeded"
                )

            per_limit = self.policy.max_tool_calls_per_tool.get(tool_name)
            if per_limit is not None:
                if self.tool_calls_by_name.get(tool_name, 0) >= per_limit:
                    raise PolicyViolation(
                        f"Tool '{tool_name}' call limit {per_limit} exceeded"
                    )

        # Graph mode: per-call validation
        if isinstance(self.policy, GraphPolicy):
            self.check_graph_step(f"tool:{tool_name}")

        self.tool_calls += 1
        self.tool_calls_by_name[tool_name] = self.tool_calls_by_name.get(tool_name, 0) + 1
        self.log("tool_call", {
            "tool": tool_name,
            "count": self.tool_calls,
            "input": str(input_data)[:200],
        })

    # ------------------------------------------------------------------
    # Token tracking
    # ------------------------------------------------------------------

    def add_token_usage(self, input_tokens: int = 0, output_tokens: int = 0, model: str = None):
        self.total_input_tokens += input_tokens or 0
        self.total_output_tokens += output_tokens or 0
        self.total_tokens += (input_tokens or 0) + (output_tokens or 0)
        self.log("token_usage", {
            "model": model,
            "input_tokens": input_tokens or 0,
            "output_tokens": output_tokens or 0,
            "running_total": self.total_tokens,
        })
        if (self.policy.max_tokens is not None
                and self.total_tokens > self.policy.max_tokens):
            raise PolicyViolation(
                f"Token limit {self.policy.max_tokens} exceeded (used {self.total_tokens})"
            )

    # ------------------------------------------------------------------
    # Graph validation
    # ------------------------------------------------------------------

    def check_graph_step(self, step: str):
        """Validate a graph transition. On bad transition: record violation, reset
        to start so the next agent turn begins cleanly, then raise PolicyViolation."""
        successors = self._adjacency.get(self.current_node, [])
        if step not in successors:
            msg = (
                f"No edge from '{self.current_node}' to '{step}' in graph. "
                f"Allowed next steps: {successors}"
            )
            self.assertion_failures.append(msg)
            self.log("graph_violation", {"message": msg})
            self._reset_for_next_turn()
            raise PolicyViolation(msg)
        self.current_node = step
        self.call_sequence.append(step)

    def advance_finish_if_complete(self):
        """Advance to 'finish' when the agent turn ends normally (LLM returned a final
        answer / chain_end fired). Only moves if there is an explicit edge to 'finish'
        from the current node, then writes the audit file and resets for the next turn."""
        if not isinstance(self.policy, GraphPolicy):
            return
        if self.current_node == "finish":
            return
        if "finish" not in self._adjacency.get(self.current_node, []):
            return
        self.current_node = "finish"
        self.call_sequence.append("finish")
        self.log("graph_node", {"node": "finish"})
        self.write()
        self._reset_for_next_turn()

    def _reset_for_next_turn(self):
        """Reset graph position and all per-turn counters so the next agent turn starts clean.
        Idempotent: no-op (and no log) if already at the start node."""
        if not isinstance(self.policy, GraphPolicy):
            return
        start = self.policy.nodes[0]
        if self.current_node == start:
            return
        self.current_node = start

        # Snapshot completed turn stats before clearing
        self.last_turn = {
            "call_sequence": list(self.call_sequence),
            "llm_calls": self.llm_calls,
            "indirect_llm_calls": self.indirect_llm_calls,
            "tool_calls": self.tool_calls,
            "tool_calls_by_name": dict(self.tool_calls_by_name),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
        }

        # Reset per-turn counters
        self.llm_calls = 0
        self.indirect_llm_calls = 0
        self.tool_calls = 0
        self.tool_calls_by_name = {}
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.call_sequence = []
        self.last_llm_mock = None
        self.last_tool_description = ""

        self.log("graph_reset", {"reset_to": start})

    def validate_graph_complete(self) -> list:
        """End-of-run check: the run must have explicitly reached 'finish'.
        Acceptable states: current_node is 'finish' (not yet reset) or the start node
        (reset after a successful turn). Any other node means the run ended mid-turn."""
        if not isinstance(self.policy, GraphPolicy):
            return []
        start = self.policy.nodes[0]
        if self.current_node in ("finish", start):
            return []
        return [
            f"Graph sequence incomplete: stopped at '{self.current_node}', "
            f"expected to reach 'finish'.\n"
            f"  Call sequence so far: {self.call_sequence}"
        ]

    # ------------------------------------------------------------------
    # Mock resolution
    # ------------------------------------------------------------------

    def find_mock(self, target: str, input_text: str) -> Optional[MockConfig]:
        """Find first mock whose target matches and match_contains (if set) is in input_text."""
        for mock in self.policy.mocks:
            if mock.target != target:
                continue
            if mock.match_contains is None or mock.match_contains in input_text:
                return mock
        return None

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------

    def run_assertions(self, target: str, check: str, value: Any):
        """Evaluate all matching assertions; failures are recorded (not raised)."""
        for assertion in self.policy.assertions:
            if assertion.target != target:
                continue
            if assertion.check != check:
                continue
            passed = _evaluate_assertion(assertion.operator, assertion.expected, value)
            if not passed:
                label = assertion.description or f"{target}.{check}"
                msg = (
                    f"Assertion FAILED [{label}]: "
                    f"{check}={repr(str(value))!r} does not satisfy "
                    f"{assertion.operator.value}({repr(assertion.expected)!r})"
                )
                self.assertion_failures.append(msg)
                self.log("assertion_failure", {"message": msg})

    # ------------------------------------------------------------------
    # Audit output
    # ------------------------------------------------------------------

    def write(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.agent_name}_audit_{timestamp}.json"
        path = Path(self.policy.audit_file).parent / filename
        print(f"[STATE] writing audit file: {path}", flush=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump({
                "trace_id": self.trace_id,
                "passed": self.passed,
                "totals": {
                    "llm_calls": self.llm_calls,
                    "indirect_llm_calls": self.indirect_llm_calls,
                    "tool_calls": self.tool_calls,
                    "tool_calls_by_name": self.tool_calls_by_name,
                    "input_tokens": self.total_input_tokens,
                    "output_tokens": self.total_output_tokens,
                    "total_tokens": self.total_tokens,
                    "call_sequence": self.call_sequence,
                },
                "assertion_failures": self.assertion_failures,
                "events": self.events,
            }, f, indent=2, ensure_ascii=False)
