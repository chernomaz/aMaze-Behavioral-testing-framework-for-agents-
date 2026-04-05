"""
System tests for the aMazeTest behavioral testing framework.
Uses fake LLM/tool classes — no real API keys required.
"""
import sys
import os
import unittest
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.tools import BaseTool, StructuredTool, tool

from amaze.policy import (
    AssertionConfig,
    AssertionOperator,
    ControlPlanePolicy,
    GraphPolicy,
    MockConfig,
)
from amaze.state import PolicyViolation, RuntimeState
from amaze.instrumentation import install


# ---------------------------------------------------------------------------
# Fake LLM and Tool helpers
# ---------------------------------------------------------------------------

class FakeLLM(BaseChatModel):
    """A BaseChatModel that raises if actually called (use mocks instead)."""

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("FakeLLM._generate should not be called — use a mock")

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("FakeLLM._stream should not be called")


def make_real_llm(return_content: str = "real response"):
    """A BaseChatModel that actually returns a configured response."""
    class RealFakeLLM(BaseChatModel):
        @property
        def _llm_type(self):
            return "real_fake"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            msg = AIMessage(content=return_content)
            return ChatResult(generations=[ChatGeneration(message=msg)])

    return RealFakeLLM()


class _FakeTool(BaseTool):
    """Reusable fake BaseTool; set name/description/real_output at instantiation."""
    name: str = "fake"
    description: str = "Fake tool"
    real_output: Optional[str] = None

    def _run(self, *args, **kwargs):
        if self.real_output is not None:
            return self.real_output
        raise RuntimeError(f"Real tool '{self.name}' should not be called — use a mock")


def make_tool(name: str, real_output: str = None):
    return _FakeTool(name=name, description=f"Fake tool {name}", real_output=real_output)


def make_cp_policy(**kwargs) -> ControlPlanePolicy:
    defaults = dict(
        allowed_tools=set(),
        max_llm_calls=None,
        max_tool_calls=None,
        max_tool_calls_per_tool={},
        max_tokens=None,
        mocks=[],
        assertions=[],
        audit_file="/tmp/amaze_test_audit.json",
    )
    defaults.update(kwargs)
    return ControlPlanePolicy(**defaults)


def make_graph_policy(**kwargs) -> GraphPolicy:
    defaults = dict(
        nodes=["agent", "finish"],
        edges=[],
        ignore_internal_llm=True,
        mocks=[],
        assertions=[],
        max_tokens=None,
        audit_file="/tmp/amaze_test_audit.json",
    )
    defaults.update(kwargs)
    return GraphPolicy(**defaults)


class PatchedContext:
    """Context manager that installs instrumentation and restores class methods on exit."""

    def __init__(self, runtime: RuntimeState):
        self.runtime = runtime
        self._saved = {}

    def __enter__(self) -> RuntimeState:
        try:
            from langgraph.pregel import Pregel
            pregel_invoke = Pregel.invoke
            pregel_ainvoke = Pregel.ainvoke
        except ImportError:
            pregel_invoke = pregel_ainvoke = None
        self._saved = {
            "llm_invoke": BaseChatModel.invoke,
            "llm_ainvoke": BaseChatModel.ainvoke,
            "tool_invoke": BaseTool.invoke,
            "tool_ainvoke": BaseTool.ainvoke,
            "structured_ainvoke": StructuredTool.__dict__.get("ainvoke"),
            "pregel_invoke": pregel_invoke,
            "pregel_ainvoke": pregel_ainvoke,
        }
        install(self.runtime)
        return self.runtime

    def __exit__(self, *args):
        BaseChatModel.invoke = self._saved["llm_invoke"]
        BaseChatModel.ainvoke = self._saved["llm_ainvoke"]
        BaseTool.invoke = self._saved["tool_invoke"]
        BaseTool.ainvoke = self._saved["tool_ainvoke"]
        if self._saved["structured_ainvoke"] is not None:
            StructuredTool.ainvoke = self._saved["structured_ainvoke"]
        try:
            from langgraph.pregel import Pregel
            if self._saved["pregel_invoke"] is not None:
                Pregel.invoke = self._saved["pregel_invoke"]
            if self._saved["pregel_ainvoke"] is not None:
                Pregel.ainvoke = self._saved["pregel_ainvoke"]
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Bug fix tests
# ---------------------------------------------------------------------------

class TestBugFixes(unittest.TestCase):

    def test_token_counters_initialized(self):
        """Bug fix: token counters must be initialized in __init__."""
        policy = make_cp_policy()
        runtime = RuntimeState(policy)
        # Should not raise AttributeError
        runtime.add_token_usage(input_tokens=10, output_tokens=5)
        self.assertEqual(runtime.total_input_tokens, 10)
        self.assertEqual(runtime.total_output_tokens, 5)
        self.assertEqual(runtime.total_tokens, 15)

    def test_enter_llm_accepts_model_kwarg(self):
        """Bug fix: enter_llm must accept 'model' keyword argument."""
        policy = make_cp_policy()
        runtime = RuntimeState(policy)
        runtime.enter_llm(model="gpt-4")
        self.assertEqual(runtime.llm_calls, 1)

    def test_enter_llm_accepts_is_indirect(self):
        policy = make_cp_policy()
        runtime = RuntimeState(policy)
        runtime.enter_llm(is_indirect=True)
        self.assertEqual(runtime.llm_calls, 0)
        self.assertEqual(runtime.indirect_llm_calls, 1)


# ---------------------------------------------------------------------------
# Control plane tests
# ---------------------------------------------------------------------------

class TestControlPlaneMode(unittest.TestCase):

    def test_llm_call_limit_enforced(self):
        policy = make_cp_policy(
            max_llm_calls=2,
            mocks=[MockConfig(target="llm", output=None, return_ai_message="ok")],
        )
        llm = FakeLLM()
        with PatchedContext(RuntimeState(policy)) as runtime:
            llm.invoke([HumanMessage(content="q1")])
            llm.invoke([HumanMessage(content="q2")])
            with self.assertRaises(PolicyViolation) as ctx:
                llm.invoke([HumanMessage(content="q3")])
        self.assertIn("LLM call limit", str(ctx.exception))

    def test_disallowed_tool_raises(self):
        policy = make_cp_policy(
            allowed_tools={"pdf_search"},
            mocks=[MockConfig(target="tool:web_search", output="blocked")],
        )
        web_search = make_tool("web_search")
        with PatchedContext(RuntimeState(policy)) as runtime:
            with self.assertRaises(PolicyViolation) as ctx:
                web_search.invoke("some query")
        self.assertIn("not in the allowed list", str(ctx.exception))

    def test_allowed_tool_passes(self):
        policy = make_cp_policy(
            allowed_tools={"pdf_search"},
            mocks=[MockConfig(target="tool:pdf_search", output="result")],
        )
        pdf_search = make_tool("pdf_search")
        with PatchedContext(RuntimeState(policy)) as runtime:
            result = pdf_search.invoke("some query")
        self.assertEqual(result, "result")

    def test_global_tool_call_limit(self):
        policy = make_cp_policy(
            max_tool_calls=2,
            mocks=[MockConfig(target="tool:search", output="r")],
        )
        search = make_tool("search")
        with PatchedContext(RuntimeState(policy)) as runtime:
            search.invoke("q1")
            search.invoke("q2")
            with self.assertRaises(PolicyViolation):
                search.invoke("q3")

    def test_per_tool_call_limit(self):
        policy = make_cp_policy(
            max_tool_calls_per_tool={"pdf_search": 2},
            mocks=[MockConfig(target="tool:pdf_search", output="r")],
        )
        pdf = make_tool("pdf_search")
        with PatchedContext(RuntimeState(policy)) as runtime:
            pdf.invoke("q1")
            pdf.invoke("q2")
            with self.assertRaises(PolicyViolation) as ctx:
                pdf.invoke("q3")
        self.assertIn("pdf_search", str(ctx.exception))

    def test_token_limit_raises(self):
        policy = make_cp_policy(max_tokens=100)
        runtime = RuntimeState(policy)
        runtime.add_token_usage(input_tokens=60, output_tokens=30)
        with self.assertRaises(PolicyViolation) as ctx:
            runtime.add_token_usage(input_tokens=20, output_tokens=0)
        self.assertIn("Token limit", str(ctx.exception))


# ---------------------------------------------------------------------------
# Graph mode tests
# ---------------------------------------------------------------------------

class TestGraphMode(unittest.TestCase):

    def _make_policy(self, nodes, edges, mocks=None):
        return make_graph_policy(nodes=nodes, edges=edges, mocks=mocks or [])

    def test_valid_sequence_passes(self):
        policy = self._make_policy(
            nodes=["agent", "llm", "tool:multiply", "finish"],
            edges=[["agent", "llm"], ["llm", "tool:multiply"], ["tool:multiply", "finish"]],
            mocks=[
                MockConfig(target="llm", return_tool_call={"tool": "multiply", "args": {"a": 3, "b": 4}}),
                MockConfig(target="tool:multiply", output="12"),
            ],
        )
        # make_real_llm used so indirect call returns a final answer (no tool_calls),
        # which triggers advance_finish_if_complete → "finish" appended.
        llm = make_real_llm("The answer is 12.")
        mul = make_tool("multiply")

        with PatchedContext(RuntimeState(policy)) as runtime:
            llm.invoke([HumanMessage(content="what is 3*4?")])
            mul.invoke({"a": 3, "b": 4})
            # indirect LLM call (tool result in history) → advances to finish
            llm.invoke([
                HumanMessage(content="what is 3*4?"),
                AIMessage(content="", tool_calls=[{"name": "multiply", "args": {"a": 3, "b": 4}, "id": "tc1", "type": "tool_call"}]),
                ToolMessage(content="12", tool_call_id="tc1"),
            ])

        failures = runtime.validate_graph_complete()
        self.assertEqual(failures, [])
        self.assertEqual(runtime.last_turn["call_sequence"], ["llm", "tool:multiply", "finish"])

    def test_wrong_order_raises_immediately(self):
        policy = self._make_policy(
            nodes=["agent", "llm", "tool:multiply", "finish"],
            edges=[["agent", "llm"], ["llm", "tool:multiply"], ["tool:multiply", "finish"]],
            mocks=[
                MockConfig(target="tool:multiply", output="12"),
                MockConfig(target="llm", return_ai_message="done"),
            ],
        )
        llm = FakeLLM()
        mul = make_tool("multiply")

        with PatchedContext(RuntimeState(policy)) as runtime:
            # tool called before LLM — records violation, resets graph, then raises
            with self.assertRaises(PolicyViolation) as ctx:
                mul.invoke({"a": 3, "b": 4})

        self.assertIn("No edge from 'agent' to 'tool:multiply'", str(ctx.exception))
        self.assertEqual(len(runtime.assertion_failures), 1)

    def test_incomplete_sequence_detected(self):
        """validate_graph_complete reports incomplete runs."""
        policy = self._make_policy(
            nodes=["agent", "llm", "tool:multiply", "finish"],
            edges=[["agent", "llm"], ["llm", "tool:multiply"], ["tool:multiply", "finish"]],
            mocks=[MockConfig(target="llm", return_tool_call={"tool": "multiply", "args": {}})],
        )
        llm = FakeLLM()

        with PatchedContext(RuntimeState(policy)) as runtime:
            llm.invoke([HumanMessage(content="go")])
            # tool:multiply never called

        failures = runtime.validate_graph_complete()
        self.assertEqual(len(failures), 1)
        self.assertIn("incomplete", failures[0])

    def test_extra_step_raises(self):
        """Calling more steps than expected raises PolicyViolation."""
        policy = self._make_policy(
            nodes=["agent", "llm", "finish"],
            edges=[["agent", "llm"], ["llm", "finish"]],
            mocks=[
                MockConfig(target="llm", return_ai_message="done"),
                MockConfig(target="tool:extra", output="x"),
            ],
        )
        llm = FakeLLM()
        extra = make_tool("extra")

        with PatchedContext(RuntimeState(policy)) as runtime:
            # LLM returns final answer → advance_finish_if_complete → finish → reset to agent
            llm.invoke([HumanMessage(content="q")])
            # Now at "agent"; calling an unexpected tool raises
            with self.assertRaises(PolicyViolation) as ctx:
                extra.invoke("something")

        self.assertIn("No edge from 'agent' to 'tool:extra'", str(ctx.exception))
        self.assertEqual(len(runtime.assertion_failures), 1)


# ---------------------------------------------------------------------------
# Mocking tests
# ---------------------------------------------------------------------------

class TestMocking(unittest.TestCase):

    def test_tool_mock_returns_configured_output(self):
        policy = make_cp_policy(mocks=[MockConfig(target="tool:multiply", output="999")])
        mul = make_tool("multiply")
        with PatchedContext(RuntimeState(policy)) as runtime:
            result = mul.invoke({"a": 3, "b": 4})
        self.assertEqual(result, "999")

    def test_llm_mock_returns_ai_message_text(self):
        policy = make_cp_policy(
            mocks=[MockConfig(target="llm", return_ai_message="mocked response")]
        )
        llm = FakeLLM()
        with PatchedContext(RuntimeState(policy)) as runtime:
            result = llm.invoke([HumanMessage(content="hello")])
        self.assertIsInstance(result, AIMessage)
        self.assertEqual(result.content, "mocked response")

    def test_llm_mock_returns_tool_call(self):
        policy = make_cp_policy(
            mocks=[MockConfig(
                target="llm",
                return_tool_call={"tool": "multiply", "args": {"a": 5, "b": 6}}
            )]
        )
        llm = FakeLLM()
        with PatchedContext(RuntimeState(policy)) as runtime:
            result = llm.invoke([HumanMessage(content="calculate 5*6")])
        self.assertIsInstance(result, AIMessage)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0]["name"], "multiply")
        self.assertEqual(result.tool_calls[0]["args"], {"a": 5, "b": 6})

    def test_mock_match_contains_filters_correctly(self):
        """match_contains restricts which inputs are mocked."""
        policy = make_cp_policy(
            mocks=[MockConfig(target="llm", match_contains="calculate", return_ai_message="42")]
        )
        real_llm = make_real_llm("real answer")
        with PatchedContext(RuntimeState(policy)) as runtime:
            # Input contains "calculate" → mock
            r1 = real_llm.invoke([HumanMessage(content="please calculate 5*6")])
            # Input doesn't contain "calculate" → real LLM
            r2 = real_llm.invoke([HumanMessage(content="what is the weather?")])

        self.assertEqual(r1.content, "42")
        self.assertEqual(r2.content, "real answer")

    def test_no_mock_calls_real_tool(self):
        """When no mock matches, the real tool is called."""
        policy = make_cp_policy()  # no mocks
        real_tool = make_tool("multiply", real_output="real_42")
        with PatchedContext(RuntimeState(policy)) as runtime:
            result = real_tool.invoke({"a": 6, "b": 7})
        self.assertEqual(result, "real_42")

    def test_tool_mock_match_contains(self):
        """Tool mock applies only when input matches."""
        policy = make_cp_policy(
            mocks=[MockConfig(target="tool:search", match_contains="python", output="Python result")]
        )
        real_search = make_tool("search", real_output="real result")
        with PatchedContext(RuntimeState(policy)) as runtime:
            r1 = real_search.invoke("python tutorial")   # matches mock
            r2 = real_search.invoke("javascript tutorial")  # no match → real
        self.assertEqual(r1, "Python result")
        self.assertEqual(r2, "real result")


# ---------------------------------------------------------------------------
# Assertion tests
# ---------------------------------------------------------------------------

class TestAssertions(unittest.TestCase):

    def test_tool_input_contains_passes(self):
        policy = make_cp_policy(
            mocks=[MockConfig(target="tool:search", output="result")],
            assertions=[AssertionConfig(
                target="tool:search", check="input",
                operator=AssertionOperator.CONTAINS, expected="python"
            )],
        )
        search = make_tool("search")
        with PatchedContext(RuntimeState(policy)) as runtime:
            search.invoke({"query": "python tutorial"})
        self.assertEqual(runtime.assertion_failures, [])

    def test_tool_input_contains_fails(self):
        policy = make_cp_policy(
            mocks=[MockConfig(target="tool:search", output="result")],
            assertions=[AssertionConfig(
                target="tool:search", check="input",
                operator=AssertionOperator.CONTAINS, expected="python"
            )],
        )
        search = make_tool("search")
        with PatchedContext(RuntimeState(policy)) as runtime:
            search.invoke({"query": "javascript tutorial"})
        self.assertEqual(len(runtime.assertion_failures), 1)
        self.assertIn("FAILED", runtime.assertion_failures[0])

    def test_tool_output_equals_passes(self):
        policy = make_cp_policy(
            mocks=[MockConfig(target="tool:calc", output="42")],
            assertions=[AssertionConfig(
                target="tool:calc", check="output",
                operator=AssertionOperator.EQUALS, expected="42"
            )],
        )
        calc = make_tool("calc")
        with PatchedContext(RuntimeState(policy)) as runtime:
            calc.invoke("6*7")
        self.assertEqual(runtime.assertion_failures, [])

    def test_llm_output_assertion(self):
        policy = make_cp_policy(
            mocks=[MockConfig(target="llm", return_ai_message="The capital is Paris.")],
            assertions=[AssertionConfig(
                target="llm", check="output",
                operator=AssertionOperator.CONTAINS, expected="Paris"
            )],
        )
        llm = FakeLLM()
        with PatchedContext(RuntimeState(policy)) as runtime:
            llm.invoke([HumanMessage(content="What is the capital of France?")])
        self.assertEqual(runtime.assertion_failures, [])

    def test_assertion_starts_with(self):
        policy = make_cp_policy(
            mocks=[MockConfig(target="tool:echo", output="Hello world")],
            assertions=[AssertionConfig(
                target="tool:echo", check="output",
                operator=AssertionOperator.STARTS_WITH, expected="Hello"
            )],
        )
        echo = make_tool("echo")
        with PatchedContext(RuntimeState(policy)) as runtime:
            echo.invoke("anything")
        self.assertEqual(runtime.assertion_failures, [])

    def test_assertion_regex(self):
        policy = make_cp_policy(
            mocks=[MockConfig(target="tool:nums", output="result: 42")],
            assertions=[AssertionConfig(
                target="tool:nums", check="output",
                operator=AssertionOperator.MATCHES_REGEX, expected=r"\d+"
            )],
        )
        nums = make_tool("nums")
        with PatchedContext(RuntimeState(policy)) as runtime:
            nums.invoke("input")
        self.assertEqual(runtime.assertion_failures, [])


# ---------------------------------------------------------------------------
# Indirect LLM detection tests
# ---------------------------------------------------------------------------

class TestIndirectLLMDetection(unittest.TestCase):

    def test_direct_call_counted_in_sequence(self):
        policy = make_graph_policy(
            nodes=["agent", "llm", "finish"],
            edges=[["agent", "llm"], ["llm", "finish"]],
            mocks=[MockConfig(target="llm", return_ai_message="hello")],
        )
        llm = FakeLLM()
        with PatchedContext(RuntimeState(policy)) as runtime:
            llm.invoke([HumanMessage(content="hi")])

        # LLM returned final answer (no tool_calls) → advance_finish_if_complete appends "finish"
        self.assertEqual(runtime.last_turn["call_sequence"], ["llm", "finish"])
        self.assertEqual(runtime.last_turn["indirect_llm_calls"], 0)
        self.assertEqual(runtime.last_turn["llm_calls"], 1)

    def test_indirect_call_not_in_sequence(self):
        """LLM call with ToolMessage in history is indirect and skips graph tracking."""
        policy = make_graph_policy(
            nodes=["agent", "llm", "finish"],
            edges=[["agent", "llm"], ["llm", "finish"]],
            mocks=[MockConfig(target="llm", return_ai_message="done")],
        )
        llm = make_real_llm("done")
        msgs_with_tool_result = [
            HumanMessage(content="what is 3*4?"),
            AIMessage(content="", tool_calls=[{
                "name": "multiply", "args": {"a": 3, "b": 4},
                "id": "tc1", "type": "tool_call"
            }]),
            ToolMessage(content="12", tool_call_id="tc1"),
        ]

        with PatchedContext(RuntimeState(policy)) as runtime:
            llm.invoke([HumanMessage(content="direct call")])  # direct
            llm.invoke(msgs_with_tool_result)                   # indirect

        # Direct call completed the turn: snapshot has 0 indirect (indirect ran in the new turn after reset)
        self.assertEqual(runtime.last_turn["call_sequence"], ["llm", "finish"])
        self.assertEqual(runtime.last_turn["indirect_llm_calls"], 0)
        self.assertEqual(runtime.last_turn["llm_calls"], 1)
        # Indirect call ran in the new (in-progress) turn
        self.assertEqual(runtime.indirect_llm_calls, 1)
        # Graph is complete (current_node reset to "agent" = start after finish)
        self.assertEqual(runtime.validate_graph_complete(), [])

    def test_indirect_call_no_graph_violation(self):
        """Indirect LLM calls don't trigger graph step checks."""
        policy = make_graph_policy(
            nodes=["agent", "llm", "tool:multiply", "finish"],
            edges=[["agent", "llm"], ["llm", "tool:multiply"], ["tool:multiply", "finish"]],
            mocks=[
                MockConfig(target="llm", return_tool_call={"tool": "multiply", "args": {"a": 1, "b": 2}}),
                MockConfig(target="tool:multiply", output="2"),
            ],
        )
        # Use a real-responding LLM for the indirect call (mocks don't apply to indirect calls)
        llm = make_real_llm("The answer is 2.")
        mul = make_tool("multiply")

        with PatchedContext(RuntimeState(policy)) as runtime:
            # Step 1: direct LLM call (agent -> llm) — mock applies
            llm.invoke([HumanMessage(content="go")])
            # Step 2: tool call (llm -> tool:multiply)
            mul.invoke({"a": 1, "b": 2})
            # Step 3: indirect LLM call (tool results) — no mock, real LLM responds, no graph advance
            llm.invoke([
                HumanMessage(content="go"),
                AIMessage(content="", tool_calls=[{"name": "multiply", "args": {}, "id": "x", "type": "tool_call"}]),
                ToolMessage(content="2", tool_call_id="x"),
            ])

        # Indirect LLM returns final answer → advance_finish_if_complete appends "finish"
        self.assertEqual(runtime.last_turn["call_sequence"], ["llm", "tool:multiply", "finish"])
        self.assertEqual(runtime.validate_graph_complete(), [])


# ---------------------------------------------------------------------------
# System tests — simulate a full LangGraph-style agent turn
# These use @tool-decorated functions (StructuredTool) and real agent message
# flows so that bugs like StructuredTool.ainvoke bypass are caught.
# ---------------------------------------------------------------------------

@tool
def search_tool(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


@tool
async def async_search_tool(query: str) -> str:
    """Async search for information."""
    return f"Async results for: {query}"


def _make_tool_call_msg(tool_name: str, args: dict, call_id: str = "tc1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{
        "name": tool_name, "args": args, "id": call_id, "type": "tool_call"
    }])


def _make_tool_result_msgs(query: str, result: str, call_id: str = "tc1") -> list:
    return [
        HumanMessage(content=query),
        _make_tool_call_msg("search_tool", {"query": query}, call_id),
        ToolMessage(content=result, tool_call_id=call_id),
    ]


class TestSystemAgentTurn(unittest.TestCase):
    """Simulate a full LangGraph agent turn: direct LLM → tool → indirect LLM → done."""

    def _policy(self, **kwargs):
        return make_graph_policy(
            nodes=["agent", "llm", "tool:search_tool", "finish"],
            edges=[
                ["agent", "llm"],
                ["llm", "tool:search_tool"],
                ["tool:search_tool", "finish"],
            ],
            **kwargs,
        )

    def test_full_sync_turn_passes(self):
        """LLM (direct, mocked) → @tool sync invoke → LLM (indirect, real) → graph complete."""
        policy = self._policy(mocks=[
            MockConfig(target="llm", return_tool_call={"tool": "search_tool", "args": {"query": "test"}}),
            MockConfig(target="tool:search_tool", output="some results"),
        ])
        llm = make_real_llm("Here are the results.")

        with PatchedContext(RuntimeState(policy)) as runtime:
            # Step 1: agent calls LLM directly
            llm.invoke([HumanMessage(content="search for test")])
            # Step 2: LangGraph tool node calls tool via invoke (sync)
            search_tool.invoke({"query": "test"})
            # Step 3: agent calls LLM with tool result (indirect)
            llm.invoke(_make_tool_result_msgs("search for test", "some results"))

        # Indirect LLM returns final answer → advance_finish_if_complete appends "finish"
        self.assertEqual(runtime.last_turn["call_sequence"], ["llm", "tool:search_tool", "finish"])
        self.assertEqual(runtime.validate_graph_complete(), [])
        self.assertEqual(runtime.last_turn["llm_calls"], 1)
        self.assertEqual(runtime.last_turn["indirect_llm_calls"], 1)
        self.assertEqual(runtime.last_turn["tool_calls"], 1)

    def test_full_async_turn_passes(self):
        """LLM (direct) → @tool async ainvoke (StructuredTool) → LLM (indirect) → complete.
        Catches the StructuredTool.ainvoke bypass bug: if ainvoke is not patched,
        enter_tool is never called and the graph stays at 'llm', causing a second
        direct LLM call to fail with 'No edge from llm to llm'."""
        import asyncio
        policy = self._policy(mocks=[
            MockConfig(target="llm", return_tool_call={"tool": "search_tool", "args": {"query": "test"}}),
            MockConfig(target="tool:search_tool", output="async results"),
        ])
        llm = make_real_llm("Here are the results.")

        async def run():
            with PatchedContext(RuntimeState(policy)) as runtime:
                # Step 1: direct LLM call
                await llm.ainvoke([HumanMessage(content="search for test")])
                # Step 2: LangGraph tool node calls @tool via ainvoke (StructuredTool path)
                await search_tool.ainvoke({"query": "test"})
                # Step 3: indirect LLM call with tool result
                await llm.ainvoke(_make_tool_result_msgs("search for test", "async results"))
            return runtime

        runtime = asyncio.run(run())
        # Indirect LLM returns final answer → advance_finish_if_complete appends "finish"
        self.assertEqual(runtime.last_turn["call_sequence"], ["llm", "tool:search_tool", "finish"])
        self.assertEqual(runtime.validate_graph_complete(), [])

    def test_mock_not_applied_to_indirect_llm(self):
        """LLM mock (return_tool_call) must NOT fire on the post-tool indirect call.
        Previously this caused an infinite tool loop: mock re-triggered the tool
        on every indirect call."""
        policy = self._policy(mocks=[
            MockConfig(target="llm", return_tool_call={"tool": "search_tool", "args": {"query": "x"}}),
            MockConfig(target="tool:search_tool", output="r"),
        ])
        llm = make_real_llm("Final answer.")

        indirect_results = []

        with PatchedContext(RuntimeState(policy)) as runtime:
            # Direct LLM call — mock fires, returns tool_call
            result1 = llm.invoke([HumanMessage(content="search for x")])
            self.assertEqual(len(result1.tool_calls), 1)

            search_tool.invoke({"query": "x"})

            # Indirect LLM call — mock must NOT fire; real LLM runs
            result2 = llm.invoke(_make_tool_result_msgs("search for x", "r"))
            indirect_results.append(result2)

        # Indirect call should return the real LLM response, not the mock tool_call
        self.assertEqual(indirect_results[0].content, "Final answer.")
        self.assertEqual(len(indirect_results[0].tool_calls), 0)
        self.assertEqual(runtime.validate_graph_complete(), [])

    def test_second_direct_llm_without_tool_raises(self):
        """If graph stays at 'llm' (e.g. tool call bypasses instrumentation),
        the next direct LLM call raises PolicyViolation immediately."""
        policy = self._policy(mocks=[
            MockConfig(target="llm", return_tool_call={"tool": "search_tool", "args": {}}),
        ])
        llm = FakeLLM()

        with PatchedContext(RuntimeState(policy)) as runtime:
            llm.invoke([HumanMessage(content="search for something")])
            # Simulate tool being called WITHOUT going through BaseTool.invoke
            # (no enter_tool called → graph stays at 'llm')
            # Second direct LLM call raises:
            with self.assertRaises(PolicyViolation) as ctx:
                llm.invoke([HumanMessage(content="search again")])

        self.assertIn("No edge from 'llm' to 'llm'", str(ctx.exception))
        self.assertEqual(len(runtime.assertion_failures), 1)

    def test_graph_complete_fails_if_tool_never_called(self):
        """If the tool is never called (mock returned tool_call but tool was skipped),
        validate_graph_complete must report the sequence as incomplete."""
        policy = self._policy(mocks=[
            MockConfig(target="llm", return_tool_call={"tool": "search_tool", "args": {}}),
        ])
        llm = FakeLLM()

        with PatchedContext(RuntimeState(policy)) as runtime:
            llm.invoke([HumanMessage(content="search for something")])
            # tool never called

        failures = runtime.validate_graph_complete()
        self.assertEqual(len(failures), 1)
        self.assertIn("incomplete", failures[0])
        self.assertIn("llm", failures[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
