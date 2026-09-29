"""LangGraph definition: a single agent with four nodes plus a human-approval interrupt.

    route --tool_use--> execute --> self_correct --> route ...
      |                    |
      |                    +--ask_clarification / refuse--> END
      +--text--> verify --fails once--> route
                   |
                   +--pending reorder--> await_approval (interrupt) --> END
                   +--------------------------------------------------> END

LangGraph earns its place for three things: explicit typed state, `interrupt`
for human approval, and checkpointing so a paused run can be resumed.
"""
from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agent.llm import LLM
from agent.nodes import make_nodes
from agent.state import AgentConfig, AgentState

# In-process checkpointer: paused runs survive until the API process restarts.
# The approval endpoint applies the state change itself either way; resuming
# the graph only completes the run's narrative. See docs/decisions.md.
CHECKPOINTER = InMemorySaver()


def build_graph(llm: LLM, cfg: AgentConfig, checkpointer=CHECKPOINTER):
    n = make_nodes(llm, cfg)
    g = StateGraph(AgentState)
    g.add_node("route", n["route"])
    g.add_node("execute", n["execute"])
    g.add_node("self_correct", n["self_correct"])
    g.add_node("verify", n["verify"])
    g.add_node("await_approval", n["await_approval"])

    g.add_edge(START, "route")
    g.add_conditional_edges("route", n["after_route"], {"execute": "execute", "verify": "verify", "end": END})
    g.add_conditional_edges("execute", n["after_execute"],
                            {"self_correct": "self_correct", "route": "route", "end": END})
    g.add_edge("self_correct", "route")
    g.add_conditional_edges("verify", n["after_verify"],
                            {"route": "route", "await_approval": "await_approval", "end": END})
    g.add_edge("await_approval", END)
    return g.compile(checkpointer=checkpointer)
