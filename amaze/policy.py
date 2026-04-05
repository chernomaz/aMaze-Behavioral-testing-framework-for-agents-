import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class PolicyMode(Enum):
    GRAPH = "graph"
    CONTROL_PLANE = "control_plane"


class AssertionOperator(Enum):
    EQUALS = "equals"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    MATCHES_REGEX = "matches_regex"


@dataclass
class MockConfig:
    target: str                          # "llm" or "tool:multiply"
    match_contains: Optional[str] = None # apply only if input contains this text (None = match all)
    output: Optional[str] = None         # for tool mocks: returned string
    return_ai_message: Optional[str] = None  # for LLM mocks: plain text AIMessage content
    return_tool_call: Optional[dict] = None  # for LLM mocks: {"tool": "...", "args": {...}}


@dataclass
class AssertionConfig:
    target: str
    check: str                # "input" or "output"
    operator: AssertionOperator
    expected: Any
    description: Optional[str] = None


@dataclass
class GraphPolicy:
    nodes: list               # ["agent", "llm", "tool:pdf_search", "finish"]
    edges: list               # [["agent", "llm"], ["llm", "tool:pdf_search"], ...]
    ignore_internal_llm: bool = True
    mocks: list = field(default_factory=list)
    assertions: list = field(default_factory=list)
    max_tokens: Optional[int] = None
    audit_file: str = "agent_audit.json"

    def adjacency(self) -> dict:
        """Returns {node: [successors]} adjacency map."""
        result = {}
        for src, dst in self.edges:
            result.setdefault(src, []).append(dst)
        return result


@dataclass
class ControlPlanePolicy:
    allowed_tools: set = field(default_factory=set)
    max_llm_calls: Optional[int] = None
    max_tool_calls: Optional[int] = None
    max_tool_calls_per_tool: dict = field(default_factory=dict)
    max_tokens: Optional[int] = None
    mocks: list = field(default_factory=list)
    assertions: list = field(default_factory=list)
    audit_file: str = "agent_audit.json"


class Policy:
    @staticmethod
    def load(path: str):
        with open(path) as f:
            data = json.load(f)

        mocks = [
            MockConfig(**m)
            for m in data.get("mocks", [])
        ]
        assertions = [
            AssertionConfig(
                operator=AssertionOperator(a["operator"]),
                **{k: v for k, v in a.items() if k != "operator"}
            )
            for a in data.get("assertions", [])
        ]

        mode = PolicyMode(data.get("mode", "control_plane"))
        if mode == PolicyMode.GRAPH:
            return GraphPolicy(
                nodes=data["nodes"],
                edges=data["edges"],
                ignore_internal_llm=data.get("ignore_internal_llm", True),
                mocks=mocks,
                assertions=assertions,
                max_tokens=data.get("max_tokens"),
                audit_file=data.get("audit_file", "agent_audit.json"),
            )
        else:
            return ControlPlanePolicy(
                allowed_tools=set(data.get("allowed_tools", [])),
                max_llm_calls=data.get("max_llm_calls"),
                max_tool_calls=data.get("max_tool_calls"),
                max_tool_calls_per_tool=data.get("max_tool_calls_per_tool", {}),
                max_tokens=data.get("max_tokens"),
                mocks=mocks,
                assertions=assertions,
                audit_file=data.get("audit_file", "agent_audit.json"),
            )
