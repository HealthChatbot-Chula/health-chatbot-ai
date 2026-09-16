from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from .analyst import append_citations_node, call_model
from .guardrails import (
    guardrail_input_node,
    guardrail_output_node,
    route_after_input_guardrail,
)
from .memory import input_node, summarize_conversation
from .models import chat_model, intent_model
from .slot_filling_graph import (
    ask_age_node,
    ask_fasting_node,
    ask_gender_node,
    ask_lab_node,
    acknowledge_profile_update_node,
    extract_info_node,
    route_after_extraction,
)
from .state import AgentState
from .tools import tools


def should_continue(state: AgentState):
    """
    Decide whether to continue calling the tools.
    """
    messages = state["messages"]
    last_message = messages[-1]
    if not getattr(last_message, "tool_calls", None):
        return "end"
    return "continue"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("input", input_node)
    graph.add_node("guardrail_input", guardrail_input_node)
    graph.add_node("extract_info_node", extract_info_node)
    graph.add_node("ask_lab_node", ask_lab_node)
    graph.add_node("acknowledge_profile_update", acknowledge_profile_update_node)
    graph.add_node("ask_fasting_node", ask_fasting_node)
    graph.add_node("ask_age_node", ask_age_node)
    graph.add_node("ask_gender_node", ask_gender_node)
    graph.add_node("our_agent", call_model)
    graph.add_node("guardrail_output", guardrail_output_node)
    graph.add_node("append_citations", append_citations_node)
    graph.add_node("summarize", summarize_conversation)

    tool_node = ToolNode(tools=tools)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("input")
    graph.add_edge("input", "guardrail_input")

    graph.add_conditional_edges(
        "guardrail_input",
        route_after_input_guardrail,
        {
            "blocked": END,
            "continue": "extract_info_node",
        },
    )

    graph.add_conditional_edges(
        "extract_info_node",
        route_after_extraction,
        {
            "ask_lab_node": "ask_lab_node",
            "acknowledge_profile_update": "acknowledge_profile_update",
            "ask_fasting_node": "ask_fasting_node",
            "ask_age_node": "ask_age_node",
            "ask_gender_node": "ask_gender_node",
            "our_agent": "our_agent",
        },
    )

    # Question nodes finish the current turn. The next user reply re-enters
    # this graph at "input" with the accumulated chat history/state.
    graph.add_edge("ask_lab_node", END)
    graph.add_edge("acknowledge_profile_update", END)
    graph.add_edge("ask_fasting_node", END)
    graph.add_edge("ask_age_node", END)
    graph.add_edge("ask_gender_node", END)

    graph.add_conditional_edges(
        "our_agent",
        should_continue,
        {
            "continue": "tools",
            "end": "guardrail_output",
        },
    )

    graph.add_edge("guardrail_output", "append_citations")
    graph.add_edge("append_citations", "summarize")
    graph.add_edge("summarize", END)
    graph.add_edge("tools", "our_agent")

    return graph.compile()
