import uuid

from langchain_core.messages import RemoveMessage

from .llm_timing import timed_llm_invoke
from .models import chat_model
from .state import AgentState


def input_node(state: AgentState):
    """
    Entry node that receives user input and initializes debugging metadata.
    """
    state["current_node"] = "input_node"
    state["steps"].append("Received user input")

    last_msg = state["messages"][-1]

    if last_msg.id is None:
        last_msg.id = str(uuid.uuid4())

    print(f"\n[1] >>> INPUT NODE: User said (ID: {last_msg.id}): {last_msg.content}")

    return state


def summarize_conversation(state: AgentState):
    """
    Summarizing both user and AI conversation, and keeping only 3 latest chat messages
    """
    messages = state["messages"]
    summary = state.get("summary", "")

    print("\n" + "=" * 50)
    print(f"[Memory Check] Current number of messages in state: {len(messages)}")
    print("=" * 50)

    # 3 Turns = 6 ข้อความ (User + AI) จะเริ่มสรุปเมื่อสะสมครบ 12 ข้อความขึ้นไป
    if len(messages) > 12:
        to_summarize = messages[:-6]
        kept_messages = messages[-6:]

        print(f"[Action] Exceeded 3 messages. Summarizing {len(to_summarize)} old messages...")

        chat_history_text = ""
        for m in to_summarize:
            role = "User" if m.type == "human" else "Assistant"
            chat_history_text += f"{role}: {m.content}\n"

        system_instruction = (
            "You summarize conversation history neutrally and as concisely as possible.\n"
            "Strict rules:\n"
            "1. Do not answer questions from the conversation.\n"
            "2. Do not provide medical advice or diagnose.\n"
            "3. Summarize from a third-person perspective.\n"
            "4. Write the summary in Thai."
        )

        if summary:
            summary_prompt = (
                f"{system_instruction}\n\n"
                f"Existing summary: {summary}\n\n"
                f"Incorporate the following new messages into that summary:\n{chat_history_text}"
            )
        else:
            summary_prompt = (
                "Summarize the following conversation concisely and clearly in Thai:\n"
                f"{chat_history_text}"
            )

        response = timed_llm_invoke(
            chat_model,
            summary_prompt,
            "graph_memory_summary",
        )
        delete_messages = [RemoveMessage(id=m.id) for m in to_summarize if m.id is not None]

        print(f"[Summary Result] Generated summary:\n>> {response.content}")
        print(f"[Status] Old messages deleted. Remaining messages: {len(messages) - len(to_summarize)}")

        print("\n[Retained Messages] Here are the 3 latest messages kept in memory:")
        for i, m in enumerate(kept_messages, 1):
            role = "User" if m.type == "human" else "Assistant"
            print(f"  {i}. {role}: {m.content}")

        print("=" * 50 + "\n")

        return {
            "summary": response.content,
            "messages": delete_messages,
        }

    print("[Action] Message count is 3 or less. Skipping summarization.")

    print("\n[Retained Messages] Here are the messages currently kept in memory:")
    for i, m in enumerate(messages, 1):
        role = "User" if m.type == "human" else "Assistant"
        print(f"  {i}. {role}: {m.content}")

    print("[4] >>> SUMMARIZE NODE: Cleaning memory...")

    return {"summary": summary}
