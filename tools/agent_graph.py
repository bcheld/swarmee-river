from __future__ import annotations

import atexit
import queue
import threading
import time
from typing import Any

from strands import tool

from swarmee_river.utils.agent_utils import create_sub_agent, extract_text, run_coroutine


def _invoke_llm_text(*, parent_agent: Any, system_prompt: str, prompt: str) -> str:
    if getattr(parent_agent, "model", None) is None:
        return ""
    agent = create_sub_agent(parent_agent=parent_agent, system_prompt=system_prompt)
    result = run_coroutine(agent.invoke_async(prompt))
    return extract_text(result)


class _AgentNode:
    def __init__(self, node_id: str, role: str, system_prompt: str) -> None:
        self.id = node_id
        self.role = role
        self.system_prompt = system_prompt
        self.neighbors: list[_AgentNode] = []
        self._queue: queue.Queue[str] = queue.Queue(maxsize=100)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def add_neighbor(self, neighbor: _AgentNode) -> None:
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

    def send(self, content: str) -> bool:
        try:
            self._queue.put_nowait(content)
            return True
        except queue.Full:
            return False

    def queue_size(self) -> int:
        return self._queue.qsize()

    def _run_loop(self, *, parent_agent: Any, poll_interval_s: float) -> None:
        while not self._stop.is_set():
            try:
                message_content = self._queue.get(timeout=poll_interval_s)
            except queue.Empty:
                continue

            try:
                response_text = _invoke_llm_text(
                    parent_agent=parent_agent,
                    system_prompt=self.system_prompt,
                    prompt=message_content,
                )
                if not response_text:
                    continue

                with self._lock:
                    neighbors = list(self.neighbors)

                for neighbor in neighbors:
                    neighbor.send(response_text)
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
        return bool(node and node.send(content))

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
            graphs = list(self._graphs.values())
            self._graphs = {}
        for graph in graphs:
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


def _tool_response(*, status: str, text: str) -> dict[str, Any]:
    return {"status": status, "content": [{"text": text}]}


def _manager_response(result: dict[str, Any], *, as_text_dump: bool = False) -> dict[str, Any]:
    text = str(result) if as_text_dump else result.get("message", "")
    return _tool_response(status=str(result.get("status", "error")), text=text)


@tool
async def agent_graph(
    action: str,
    graph_id: str | None = None,
    topology: dict[str, Any] | None = None,
    message: dict[str, Any] | None = None,
    agent: Any | None = None,
) -> dict[str, Any]:
    """
    Interrupt-friendly `agent_graph` replacement.

    This overrides the legacy Strands Tools `agent_graph` (which uses non-daemon ThreadPoolExecutor workers) with
    a daemon-thread implementation that won't hang the process on exit and is safe to cancel while executing.
    """
    action = (action or "").strip().lower()
    if action not in {"create", "stop", "stop_all", "message", "status", "list"}:
        return _tool_response(status="error", text=f"Unknown action: {action}")

    if action == "list":
        return _manager_response(_MANAGER.list(), as_text_dump=True)

    if action == "stop_all":
        return _manager_response(_MANAGER.stop_all())

    if not graph_id or not graph_id.strip():
        return _tool_response(status="error", text="graph_id is required.")

    if action == "create":
        if topology is None:
            return _tool_response(status="error", text="topology is required for create action.")
        if agent is None:
            return _tool_response(status="error", text="agent context is required for create action.")
        return _manager_response(_MANAGER.create(graph_id, topology, parent_agent=agent))

    if action == "stop":
        return _manager_response(_MANAGER.stop(graph_id))

    if action == "message":
        if message is None:
            return _tool_response(status="error", text="message is required for message action.")
        return _manager_response(_MANAGER.send(graph_id, message))

    if action == "status":
        return _manager_response(_MANAGER.status(graph_id), as_text_dump=True)

    return _tool_response(status="error", text="Unsupported operation.")
