from __future__ import annotations

import atexit
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from strands import tool
from strands.types.tools import ToolResult, ToolUse

try:
    from strands_tools.use_llm import use_llm as _use_llm

    _HAS_USE_LLM = True
except Exception:  # pragma: no cover
    _use_llm = None  # type: ignore[assignment]
    _HAS_USE_LLM = False


def _null_callback_handler(**_kwargs: Any) -> None:
    return None


def _extract_use_llm_text(result: ToolResult | dict[str, Any]) -> str:
    for item in result.get("content", []) if isinstance(result, dict) else []:
        if isinstance(item, dict):
            text_value = item.get("text")
            if not isinstance(text_value, str):
                continue
            text = text_value
            if text.startswith("Response:"):
                return text[len("Response:") :].strip()
            return text.strip()
    return ""


@dataclass
class _GraphMessage:
    sender: str
    content: str


class _AgentNode:
    def __init__(self, node_id: str, role: str, system_prompt: str) -> None:
        self.id = node_id
        self.role = role
        self.system_prompt = system_prompt
        self.neighbors: list["_AgentNode"] = []
        self._queue: queue.Queue[_GraphMessage] = queue.Queue(maxsize=100)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def add_neighbor(self, neighbor: "_AgentNode") -> None:
        with self._lock:
            if neighbor not in self.neighbors:
                self.neighbors.append(neighbor)

    def start(self, *, parent_agent: Any, poll_interval_s: float = 0.1) -> None:
        if self._thread is not None:
            return

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            kwargs={"parent_agent": parent_agent, "poll_interval_s": poll_interval_s},
            daemon=True,
            name=f"agent-graph-node:{self.id}",
        )
        self._thread.start()

    def stop(self, *, join_timeout_s: float = 0.5) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout_s)
            self._thread = None

    def send(self, sender: str, content: str) -> bool:
        try:
            self._queue.put_nowait(_GraphMessage(sender=sender, content=content))
            return True
        except queue.Full:
            return False

    def queue_size(self) -> int:
        return self._queue.qsize()

    def _run_loop(self, *, parent_agent: Any, poll_interval_s: float) -> None:
        while not self._stop.is_set():
            try:
                message = self._queue.get(timeout=poll_interval_s)
            except queue.Empty:
                continue

            try:
                tool_use: ToolUse = {
                    "toolUseId": str(uuid.uuid4()),
                    "input": {"system_prompt": self.system_prompt, "prompt": message.content},
                    "name": "use_llm",
                }
                result = _use_llm(
                    tool_use,
                    agent=parent_agent,
                    callback_handler=_null_callback_handler,
                    trace_attributes=getattr(parent_agent, "trace_attributes", None),
                )
                response_text = _extract_use_llm_text(result)
                if not response_text:
                    continue

                with self._lock:
                    neighbors = list(self.neighbors)

                for neighbor in neighbors:
                    neighbor.send(self.id, response_text)
            except Exception:
                # Best-effort background processing: never crash the host process.
                time.sleep(poll_interval_s)


class _AgentGraph:
    def __init__(self, graph_id: str, topology_type: str, *, parent_agent: Any) -> None:
        self.graph_id = graph_id
        self.topology_type = topology_type
        self._parent_agent = parent_agent
        self._nodes: dict[str, _AgentNode] = {}
        self._lock = threading.Lock()

    def add_node(self, node_id: str, role: str, system_prompt: str) -> None:
        with self._lock:
            self._nodes[node_id] = _AgentNode(node_id=node_id, role=role, system_prompt=system_prompt)

    def add_edge(self, from_id: str, to_id: str) -> None:
        with self._lock:
            if from_id not in self._nodes or to_id not in self._nodes:
                return
            self._nodes[from_id].add_neighbor(self._nodes[to_id])
            if self.topology_type == "mesh":
                self._nodes[to_id].add_neighbor(self._nodes[from_id])

    def start(self) -> None:
        with self._lock:
            nodes = list(self._nodes.values())
        for node in nodes:
            node.start(parent_agent=self._parent_agent)

    def stop(self) -> None:
        with self._lock:
            nodes = list(self._nodes.values())
        for node in nodes:
            node.stop()

    def send_message(self, target_id: str, content: str) -> bool:
        with self._lock:
            node = self._nodes.get(target_id)
        if node is None:
            return False
        return node.send("user", content)

    def status(self) -> dict[str, Any]:
        with self._lock:
            nodes = list(self._nodes.values())
            return {
                "graph_id": self.graph_id,
                "topology": self.topology_type,
                "nodes": [
                    {
                        "id": n.id,
                        "role": n.role,
                        "neighbors": [x.id for x in n.neighbors],
                        "queue_size": n.queue_size(),
                    }
                    for n in nodes
                ],
            }


class _AgentGraphManager:
    def __init__(self) -> None:
        self._graphs: dict[str, _AgentGraph] = {}
        self._lock = threading.Lock()

    def create(self, graph_id: str, topology: dict[str, Any], *, parent_agent: Any) -> dict[str, Any]:
        with self._lock:
            if graph_id in self._graphs:
                return {"status": "error", "message": f"Graph {graph_id} already exists"}

            topology_type = str(topology.get("type") or "").strip() or "mesh"
            nodes = topology.get("nodes") or []
            edges = topology.get("edges") or []

            graph = _AgentGraph(graph_id, topology_type, parent_agent=parent_agent)
            for node_def in nodes:
                node_id = str(node_def.get("id") or "").strip()
                system_prompt = str(node_def.get("system_prompt") or "").strip()
                if not node_id:
                    return {"status": "error", "message": "Each topology.nodes[] entry must include a non-empty 'id'."}
                if not system_prompt:
                    return {
                        "status": "error",
                        "message": f"Node {node_id!r} is missing a non-empty 'system_prompt'.",
                    }
                graph.add_node(
                    node_id=node_id,
                    role=str(node_def.get("role", "")),
                    system_prompt=system_prompt,
                )
            for edge in edges:
                graph.add_edge(str(edge.get("from")), str(edge.get("to")))

            graph.start()
            self._graphs[graph_id] = graph
            return {"status": "success", "message": f"Graph {graph_id} created and started"}

    def stop(self, graph_id: str) -> dict[str, Any]:
        with self._lock:
            graph = self._graphs.pop(graph_id, None)
        if graph is None:
            return {"status": "error", "message": f"Graph {graph_id} not found"}
        graph.stop()
        return {"status": "success", "message": f"Graph {graph_id} stopped and removed"}

    def stop_all(self) -> dict[str, Any]:
        with self._lock:
            graphs = list(self._graphs.items())
            self._graphs = {}
        for _graph_id, graph in graphs:
            graph.stop()
        return {"status": "success", "message": f"Stopped {len(graphs)} graphs"}

    def send(self, graph_id: str, message: dict[str, Any]) -> dict[str, Any]:
        target = str(message.get("target") or "").strip()
        content = str(message.get("content") or "")
        if not target:
            return {"status": "error", "message": "message.target is required"}

        with self._lock:
            graph = self._graphs.get(graph_id)
        if graph is None:
            return {"status": "error", "message": f"Graph {graph_id} not found"}

        ok = graph.send_message(target, content)
        if not ok:
            return {"status": "error", "message": f"Target node {target} not found or queue full"}
        return {"status": "success", "message": f"Message sent to node {target} in graph {graph_id}"}

    def status(self, graph_id: str) -> dict[str, Any]:
        with self._lock:
            graph = self._graphs.get(graph_id)
        if graph is None:
            return {"status": "error", "message": f"Graph {graph_id} not found"}
        return {"status": "success", "data": graph.status()}

    def list(self) -> dict[str, Any]:
        with self._lock:
            data = [
                {"graph_id": graph_id, "topology": graph.topology_type, "node_count": len(graph.status()["nodes"])}
                for graph_id, graph in self._graphs.items()
            ]
        return {"status": "success", "data": data}


_MANAGER = _AgentGraphManager()
atexit.register(lambda: _MANAGER.stop_all())


@tool
async def agent_graph(
    action: str,
    graph_id: str | None = None,
    topology: dict[str, Any] | None = None,
    message: dict[str, Any] | None = None,
    agent: Optional[Any] = None,
) -> dict[str, Any]:
    """
    Interrupt-friendly `agent_graph` replacement.

    This overrides the legacy Strands Tools `agent_graph` (which uses non-daemon ThreadPoolExecutor workers) with
    a daemon-thread implementation that won't hang the process on exit and is safe to cancel while executing.
    """
    action = (action or "").strip().lower()
    if action not in {"create", "stop", "stop_all", "message", "status", "list"}:
        return {"status": "error", "content": [{"text": f"Unknown action: {action}"}]}

    if not _HAS_USE_LLM:
        return {
            "status": "error",
            "content": [{"text": "agent_graph requires `strands_tools.use_llm` to be installed and importable."}],
        }

    if action == "list":
        result = _MANAGER.list()
        return {"status": result["status"], "content": [{"text": str(result)}]}

    if action == "stop_all":
        result = _MANAGER.stop_all()
        return {"status": result["status"], "content": [{"text": result.get("message", "")}]}

    if not graph_id or not graph_id.strip():
        return {"status": "error", "content": [{"text": "graph_id is required."}]}

    if action == "create":
        if topology is None:
            return {"status": "error", "content": [{"text": "topology is required for create action."}]}
        if agent is None:
            return {"status": "error", "content": [{"text": "agent context is required for create action."}]}
        result = _MANAGER.create(graph_id, topology, parent_agent=agent)
        return {"status": result["status"], "content": [{"text": result.get("message", "")}]}

    if action == "stop":
        result = _MANAGER.stop(graph_id)
        return {"status": result["status"], "content": [{"text": result.get("message", "")}]}

    if action == "message":
        if message is None:
            return {"status": "error", "content": [{"text": "message is required for message action."}]}
        result = _MANAGER.send(graph_id, message)
        return {"status": result["status"], "content": [{"text": result.get("message", "")}]}

    if action == "status":
        result = _MANAGER.status(graph_id)
        return {"status": result["status"], "content": [{"text": str(result)}]}

    return {"status": "error", "content": [{"text": "Unsupported operation."}]}
