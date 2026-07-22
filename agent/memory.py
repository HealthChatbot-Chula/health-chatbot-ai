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
            "คุณคือผู้ช่วยที่มีหน้าที่สรุปประวัติการสนทนาอย่างเป็นกลาง "
            "กรุณาเขียนสรุปเนื้อหาที่พูดคุยกันให้กระชับที่สุด\n"
            "กฎสำคัญที่ต้องปฏิบัติตามอย่างเคร่งครัด:\n"
            "1. ห้ามตอบคำถามที่อยู่ในบทสนทนา\n"
            "2. ห้ามให้คำแนะนำทางการแพทย์หรือวินิจฉัยโรคเด็ดขาด\n"
            "3. ให้สรุปในมุมมองบุคคลที่สาม (เช่น 'ผู้ใช้สอบถามเกี่ยวกับ...', 'ผู้ช่วยได้อธิบายเรื่อง...')"
        )

        if summary:
            summary_prompt = (
                f"{system_instruction}\n\n"
                f"สรุปเดิม: {summary}\n\n"
                f"นำข้อความใหม่เหล่านี้ไปสรุปเพิ่มรวมกับสรุปเดิม:\n{chat_history_text}"
            )
        else:
            summary_prompt = f"กรุณาสรุปเนื้อหาการสนทนาต่อไปนี้ให้กระชับและเข้าใจง่าย:\n{chat_history_text}"

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
