import functools
import uuid

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel


def install(runtime):
    _patch_llm(runtime)
    _patch_tools(runtime)
    _patch_agent(runtime)


# ---------------------------------------------------------------------------
# Agent-level callback handler (chain_end = finish, chain_error = reset)
# ---------------------------------------------------------------------------

class _GraphCallbackHandler(BaseCallbackHandler):
    """Hooks into the top-level agent chain to detect turn completion or error."""

    def __init__(self, runtime):
        super().__init__()
        self.runtime = runtime

    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
        if parent_run_id is None:          # top-level chain finished normally
            self.runtime.advance_finish_if_complete()

    def on_chain_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        if parent_run_id is None:          # top-level chain raised → reset for next turn
            self.runtime._reset_for_next_turn()


def _patch_agent(runtime):
    """Patch Pregel (parent of CompiledStateGraph / create_react_agent result) to inject
    the graph callback handler. This gives us chain_end and chain_error events for every
    top-level agent.invoke / agent.ainvoke call."""
    try:
        from langgraph.pregel import Pregel
    except ImportError:
        return  # LangGraph not installed — skip

    handler = _GraphCallbackHandler(runtime)
    orig_invoke = Pregel.invoke
    orig_ainvoke = Pregel.ainvoke

    @functools.wraps(orig_invoke)
    def patched_invoke(self, input, config=None, **kwargs):
        config = dict(config or {})
        cbs = list(config.get("callbacks") or [])
        if handler not in cbs:
            cbs.append(handler)
        config["callbacks"] = cbs
        try:
            return orig_invoke(self, input, config, **kwargs)
        except Exception:
            runtime._reset_for_next_turn()
            raise

    @functools.wraps(orig_ainvoke)
    async def patched_ainvoke(self, input, config=None, **kwargs):
        config = dict(config or {})
        cbs = list(config.get("callbacks") or [])
        if handler not in cbs:
            cbs.append(handler)
        config["callbacks"] = cbs
        try:
            return await orig_ainvoke(self, input, config, **kwargs)
        except Exception:
            runtime._reset_for_next_turn()
            raise

    Pregel.invoke = patched_invoke
    Pregel.ainvoke = patched_ainvoke


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_indirect_llm_call(input_arg) -> bool:
    """Return True if the LLM input contains a ToolMessage (indirect / tool-result call)."""
    msgs = []
    if isinstance(input_arg, list):
        msgs = input_arg
    elif isinstance(input_arg, dict) and "messages" in input_arg:
        msgs = input_arg["messages"]
    elif hasattr(input_arg, "messages"):
        msgs = input_arg.messages
    return any(isinstance(m, ToolMessage) for m in msgs)


def _input_to_text(input_arg) -> str:
    """Extract only HumanMessage content for mock matching (excludes system prompt)."""
    from langchain_core.messages import HumanMessage as _HumanMessage
    msgs = None
    if isinstance(input_arg, list):
        msgs = input_arg
    elif hasattr(input_arg, "messages"):
        msgs = input_arg.messages
    if msgs is not None:
        human_texts = [
            m.content for m in msgs if isinstance(m, _HumanMessage)
        ]
        return " ".join(human_texts)
    if isinstance(input_arg, str):
        return input_arg
    return str(input_arg)


def _build_llm_mock_response(mock) -> AIMessage:
    """Build an AIMessage from a MockConfig."""
    if mock.return_tool_call:
        tc = mock.return_tool_call
        return AIMessage(
            content="",
            tool_calls=[{
                "name": tc["tool"],
                "args": tc.get("args", {}),
                "id": str(uuid.uuid4()),
                "type": "tool_call",
            }],
        )
    if mock.return_ai_message is not None:
        return AIMessage(content=mock.return_ai_message)
    return AIMessage(content="")


def _extract_tool_args(input_arg):
    """Normalize BaseTool input to its args dict or original value."""
    if isinstance(input_arg, dict) and input_arg.get("type") == "tool_call":
        return input_arg.get("args", {})
    return input_arg


def _wrap_tool_mock_output(raw_output, input_arg, tool_name: str):
    """Return mock output in the format the caller expects.

    When called via LangGraph's ToolNode the input is a tool_call dict
    and BaseTool.invoke/ainvoke normally returns a ToolMessage.
    Returning a plain str in that case triggers LangGraph's
    'returned unexpected type' error (LangGraph 1.x+).
    """
    content = str(raw_output) if raw_output is not None else ""
    if isinstance(input_arg, dict) and input_arg.get("type") == "tool_call":
        from langchain_core.messages import ToolMessage as _ToolMessage
        return _ToolMessage(
            content=content,
            tool_call_id=input_arg.get("id", ""),
            name=tool_name,
        )
    return content


def _extract_usage_from_result(result):
    """Return (input_tokens, output_tokens) from an AIMessage."""
    usage = getattr(result, "usage_metadata", None)
    if isinstance(usage, dict):
        inp = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
        out = usage.get("output_tokens") or usage.get("completion_tokens") or 0
        if inp or out:
            return inp, out

    meta = getattr(result, "response_metadata", None)
    if isinstance(meta, dict):
        tu = meta.get("token_usage", {})
        if isinstance(tu, dict):
            inp = tu.get("prompt_tokens") or tu.get("input_tokens") or 0
            out = tu.get("completion_tokens") or tu.get("output_tokens") or 0
            if inp or out:
                return inp, out
    return 0, 0


def _build_indirect_hint(runtime) -> str:
    """Build a context hint string for indirect LLM calls when a mock redirected the tool.

    When the LLM mock replaced the user query with a different tool call (e.g. changed
    "bitcoin price today" to pdf_search("Scenarios in humanitarian data management")),
    the real LLM on the indirect call sees the original user query and may try to call
    additional tools to satisfy it. Injecting the tool description and mock query tells
    the LLM what actually happened so it can return a final answer instead.
    """
    mock = runtime.last_llm_mock
    if not mock or not mock.return_tool_call:
        return ""
    tc = mock.return_tool_call
    tool_name = tc.get("tool", "")
    query = tc.get("args", {}).get("query", "")
    desc = runtime.last_tool_description

    parts = []
    if desc:
        parts.append(f"Tool used: {tool_name} — {desc}")
    else:
        parts.append(f"Tool used: {tool_name}")
    if query:
        parts.append(f"Query sent to tool: \"{query}\"")
    parts.append("The tool results are above. Provide a final answer based on them. Do not call any additional tools.")
    return " ".join(parts)


def _inject_hint(input_arg, hint: str, mock_query: str):
    """Prepare messages for the indirect LLM call:
    - Replace HumanMessage content with the mock query so the LLM never sees
      the original user text (e.g. 'bitcoin price today').
    - Prepend a SystemMessage hint after any existing system prompt.
    """
    from langchain_core.messages import HumanMessage as _HumanMessage

    hint_msg = SystemMessage(content=hint)

    def _rewrite(msgs: list) -> list:
        result = []
        for m in msgs:
            if isinstance(m, _HumanMessage) and mock_query:
                # Replace original user text with the mock query string
                result.append(_HumanMessage(content=mock_query))
            else:
                result.append(m)
        # Insert hint after an existing leading SystemMessage if present
        insert_at = 0
        if result and isinstance(result[0], SystemMessage):
            insert_at = 1
        result.insert(insert_at, hint_msg)
        return result

    if isinstance(input_arg, list):
        return _rewrite(input_arg)
    if isinstance(input_arg, dict) and "messages" in input_arg:
        return {**input_arg, "messages": _rewrite(input_arg["messages"])}
    if hasattr(input_arg, "messages"):
        return _rewrite(list(input_arg.messages))
    return input_arg


# ---------------------------------------------------------------------------
# LLM patching  (BaseChatModel covers ChatOpenAI, ChatOllama, etc.)
# ---------------------------------------------------------------------------

def _patch_llm(runtime):
    orig_invoke = BaseChatModel.invoke
    orig_ainvoke = BaseChatModel.ainvoke

    @functools.wraps(orig_invoke)
    def patched_invoke(self, input_arg, *args, **kwargs):
        model_name = getattr(self, "model_name", None) or getattr(self, "model", "unknown")
        indirect = _is_indirect_llm_call(input_arg)

        runtime.enter_llm(model=model_name, is_indirect=indirect)

        input_text = _input_to_text(input_arg)

        # Assertions only for direct calls
        if not indirect:
            runtime.run_assertions("llm", "input", input_text)

        # Mocks apply to direct calls only; indirect calls (post-tool-result) hit the real LLM
        if not indirect:
            mock = runtime.find_mock("llm", input_text)
            if mock is not None:
                runtime.last_llm_mock = mock  # remember for indirect hint injection
                result = _build_llm_mock_response(mock)
                runtime.run_assertions("llm", "output", result.content)
                if not getattr(result, "tool_calls", None):
                    runtime.advance_finish_if_complete()
                return result

        # For indirect calls, replace the original user query with the mock query and
        # inject a hint so the real LLM understands the mock context and returns a final answer.
        if indirect:
            hint = _build_indirect_hint(runtime)
            if hint:
                mock_query = (runtime.last_llm_mock.return_tool_call or {}).get("args", {}).get("query", "")
                input_arg = _inject_hint(input_arg, hint, mock_query)

        try:
            result = orig_invoke(self, input_arg, *args, **kwargs)
        except Exception:
            runtime._reset_for_next_turn()
            raise

        inp, out = _extract_usage_from_result(result)
        runtime.add_token_usage(input_tokens=inp, output_tokens=out, model=model_name)

        if not indirect:
            runtime.run_assertions("llm", "output", result.content)

        # If the LLM returned a final answer (no tool_calls), the agent turn is done.
        # advance_finish_if_complete records 'finish' and resets for the next turn.
        # This also fires for unit tests where no Pregel chain wraps the calls.
        if not getattr(result, "tool_calls", None):
            runtime.advance_finish_if_complete()

        return result

    @functools.wraps(orig_ainvoke)
    async def patched_ainvoke(self, input_arg, *args, **kwargs):
        model_name = getattr(self, "model_name", None) or getattr(self, "model", "unknown")
        indirect = _is_indirect_llm_call(input_arg)

        runtime.enter_llm(model=model_name, is_indirect=indirect)

        input_text = _input_to_text(input_arg)

        if not indirect:
            runtime.run_assertions("llm", "input", input_text)

        # Mocks apply to direct calls only; indirect calls (post-tool-result) hit the real LLM
        if not indirect:
            mock = runtime.find_mock("llm", input_text)
            if mock is not None:
                runtime.last_llm_mock = mock  # remember for indirect hint injection
                result = _build_llm_mock_response(mock)
                runtime.run_assertions("llm", "output", result.content)
                if not getattr(result, "tool_calls", None):
                    runtime.advance_finish_if_complete()
                return result

        # For indirect calls, replace the original user query with the mock query and
        # inject a hint so the real LLM understands the mock context and returns a final answer.
        if indirect:
            hint = _build_indirect_hint(runtime)
            if hint:
                mock_query = (runtime.last_llm_mock.return_tool_call or {}).get("args", {}).get("query", "")
                input_arg = _inject_hint(input_arg, hint, mock_query)

        try:
            result = await orig_ainvoke(self, input_arg, *args, **kwargs)
        except Exception:
            runtime._reset_for_next_turn()
            raise

        inp, out = _extract_usage_from_result(result)
        runtime.add_token_usage(input_tokens=inp, output_tokens=out, model=model_name)

        if not indirect:
            runtime.run_assertions("llm", "output", result.content)

        if not getattr(result, "tool_calls", None):
            runtime.advance_finish_if_complete()

        return result

    BaseChatModel.invoke = patched_invoke
    BaseChatModel.ainvoke = patched_ainvoke


# ---------------------------------------------------------------------------
# Tool patching
# ---------------------------------------------------------------------------

def _patch_tools(runtime):
    orig_invoke = BaseTool.invoke
    orig_ainvoke = BaseTool.ainvoke

    @functools.wraps(orig_invoke)
    def patched_invoke(self, input_arg, *args, **kwargs):
        tool_name = self.name
        target = f"tool:{tool_name}"
        args_data = _extract_tool_args(input_arg)
        input_text = str(args_data)

        runtime.last_tool_description = getattr(self, "description", "")
        runtime.run_assertions(target, "input", input_text)
        runtime.enter_tool(tool_name, args_data)

        mock = runtime.find_mock(target, input_text)
        if mock is not None:
            content = str(mock.output) if mock.output is not None else ""
            runtime.run_assertions(target, "output", content)
            return _wrap_tool_mock_output(mock.output, input_arg, tool_name)

        result = orig_invoke(self, input_arg, *args, **kwargs)
        runtime.run_assertions(target, "output", result)
        return result

    @functools.wraps(orig_ainvoke)
    async def patched_ainvoke(self, input_arg, *args, **kwargs):
        tool_name = self.name
        target = f"tool:{tool_name}"
        args_data = _extract_tool_args(input_arg)
        input_text = str(args_data)

        runtime.last_tool_description = getattr(self, "description", "")
        runtime.run_assertions(target, "input", input_text)
        runtime.enter_tool(tool_name, args_data)

        mock = runtime.find_mock(target, input_text)
        if mock is not None:
            content = str(mock.output) if mock.output is not None else ""
            runtime.run_assertions(target, "output", content)
            return _wrap_tool_mock_output(mock.output, input_arg, tool_name)

        result = await orig_ainvoke(self, input_arg, *args, **kwargs)
        runtime.run_assertions(target, "output", result)
        return result

    BaseTool.invoke = patched_invoke
    BaseTool.ainvoke = patched_ainvoke
    # StructuredTool (created by @tool decorator) overrides ainvoke in its own __dict__,
    # so patching BaseTool.ainvoke alone doesn't cover async tool calls. Patch it explicitly.
    if "ainvoke" in StructuredTool.__dict__:
        StructuredTool.ainvoke = patched_ainvoke