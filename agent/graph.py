from dotenv import load_dotenv 
import os
import json 
import tempfile
from langchain_core.messages import BaseMessage, RemoveMessage # The foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from .state import AgentState
from .rag_utils import retrieve_context
from .slot_filling_graph import (
    ask_age_node,
    ask_fasting_node,
    ask_gender_node,
    ask_lab_node,
    extract_info_node,
    route_after_extraction,
    _updates_from_pending_slot,
)
from typing import Optional, Tuple
import uuid

load_dotenv()

gcp_secret = os.environ.get("GCP_CREDS_JSON")

if gcp_secret:
    try:
        # 2. สร้างไฟล์ชั่วคราว (Temp File) ระบบจะลบทิ้งอัตโนมัติเมื่อแอปปิด
        fd, temp_path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, 'w') as f:
            f.write(gcp_secret)
        
        # 3. สั่งให้ Google SDK อ่านกุญแจจากไฟล์ชั่วคราวนี้
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path
        
        # ดึง Project ID มาใช้
        creds_dict = json.loads(gcp_secret)
        os.environ["GOOGLE_CLOUD_PROJECT"] = creds_dict.get("project_id", "")
        
        print("✅ Secure Mode: โหลด Vertex AI Credentials สำเร็จ!")
    except Exception as e:
        print(f"❌ Error loading credentials: {e}")
else:
    # ถ้ารันในคอมตัวเอง (Local) แล้วไม่ได้ตั้ง GCP_CREDS_JSON ไว้
    # ระบบจะข้ามไปใช้ GOOGLE_APPLICATION_CREDENTIALS ในไฟล์ .env ของคุณตามปกติครับ
    print("⚠️ Local Mode: ใช้ Credentials จากไฟล์ .env")


@tool
def add(a: int, b:int):
    """This is an addition function that adds 2 numbers together"""

    return a + b 

@tool
def subtract(a: int, b: int):
    """Subtraction function"""
    return a - b

@tool
def multiply(a: int, b: int):
    """Multiplication function"""
    return a * b

# tools = [add, subtract, multiply]
tools = []

# model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite").bind_tools(tools)

FASTING_FOLLOWUP_LABS = {
    "FBS",
    "Glucose",
    "Total Cholesterol",
    "HDL",
    "LDL",
    "Triglycerides",
}

intent_model = ChatVertexAI(
    model="gemini-2.5-flash-lite",
    temperature=0
)

chat_model = ChatVertexAI(
    model="gemini-2.5-flash"
)


# ----- Function -----

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

    # Log the current number of messages in the state
    print("\n" + "="*50)
    print(f"[Memory Check] Current number of messages in state: {len(messages)}")
    print("="*50)

    # 3 Turns = 6 ข้อความ (User + AI) จะเริ่มสรุปเมื่อสะสมครบ 12 ข้อความขึ้นไป
    if len(messages) > 12:
        # Extracting: keeping only 6 last messages, the others will be summarized
        to_summarize = messages[:-6]
        kept_messages = messages[-6:] # Extract the 6 messages we are keeping
        
        # Log the action being taken
        print(f"[Action] Exceeded 3 messages. Summarizing {len(to_summarize)} old messages...")
        
        # Creating prompt for summarizing
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

        # Calling LLM to summarize the chat
        response = chat_model.invoke(summary_prompt)
        
        # Create a list of RemoveMessage objects to prune the state.
        delete_messages = [RemoveMessage(id=m.id) for m in to_summarize if m.id is not None]

        # Log the summarization result
        print(f"[Summary Result] Generated summary:\n>> {response.content}")
        print(f"[Status] Old messages deleted. Remaining messages: {len(messages) - len(to_summarize)}")
        
        # Log the exact 3 messages that are being kept
        print("\n[Retained Messages] Here are the 3 latest messages kept in memory:")
        for i, m in enumerate(kept_messages, 1):
            role = "User" if m.type == "human" else "Assistant"
            print(f"  {i}. {role}: {m.content}")
            
        print("="*50 + "\n")

        return {
            "summary": response.content,
            "messages": delete_messages
        }
    
    # Log when no summarization is needed
    print("[Action] Message count is 3 or less. Skipping summarization.")
    
    # Log the messages currently kept in memory (which is 3 or less)
    print("\n[Retained Messages] Here are the messages currently kept in memory:")
    for i, m in enumerate(messages, 1):
        role = "User" if m.type == "human" else "Assistant"
        print(f"  {i}. {role}: {m.content}")
        
    print(f"[4] >>> SUMMARIZE NODE: Cleaning memory...")
    
    return {"summary": summary}

def guardrail_input_node(state: AgentState):
    """
    ตรวจสอบ input ของ user ก่อนเข้าระบบหลัก
    - ALLOW: คำถามสุขภาพ, แปรผลแลป, การทักทาย, บริบทอื่นๆที่เกี่ยวข้อง
    - BLOCK: ไม่เกี่ยวกับสุขภาพเลย เช่น เกม, การเมือง, ความบันเทิง
    """
    state["current_node"] = "guardrail_input"
    state["steps"].append("guardrail_input")

    last_user_message = state["messages"][-1].content

    print(f"[1.5] 🛡️ INPUT GUARDRAIL: Checking relevance...")

    pending_slot_updates = _updates_from_pending_slot(state, last_user_message)
    if pending_slot_updates:
        print(
            "    ✅ ALLOWED: user is answering the pending slot "
            f"{state.get('pending_slot')!r}"
        )
        return {
            "blocked": False,
            "current_node": "guardrail_input",
            "steps": state.get("steps", []) + ["guardrail_input_pending_slot_allowed"],
        }

    guard_prompt = (
        "คุณคือระบบกรองคำถามของแอปพลิเคชันสุขภาพ\n"
        "หน้าที่: ตัดสินว่าข้อความของผู้ใช้ควรได้รับการประมวลผลต่อหรือไม่\n\n"
        "--- เกณฑ์การตัดสิน ---\n"
        "ALLOW (อนุญาต):\n"
        "- คำถามเกี่ยวกับอาการ โรค สุขภาพทั่วไป\n"
        "- การส่งค่าผลตรวจแลป หรือขอแปลผล\n"
        "- การทักทายทั่วไป เช่น สวัสดี, ขอบคุณ, ลาก่อน\n"
        "- คำถามเกี่ยวกับการใช้งานระบบนี้\n"
        "- ข้อความที่กำกวม ไม่แน่ใจ หรืออาจเกี่ยวกับสุขภาพได้\n\n"
        "BLOCK (ปฏิเสธ) เฉพาะเมื่อชัดเจนว่าไม่เกี่ยวกับสุขภาพเลย:\n"
        "- เกม, ภาพยนตร์, ดนตรี, ความบันเทิง\n"
        "- การเมือง, ข่าวสาร, กีฬา\n"
        "- การเขียนโค้ด, เทคโนโลยี\n"
        "- การทำอาหาร, ท่องเที่ยว, ช้อปปิ้ง\n\n"
        "ตอบเป็น JSON เท่านั้น ห้ามมีข้อความอื่น:\n"
        '{"action": "ALLOW" หรือ "BLOCK", "reason": "เหตุผลสั้นๆ 1 ประโยค"}\n\n'
        f"ข้อความของผู้ใช้: {last_user_message}"
    )

    result = intent_model.invoke(guard_prompt).content.strip()

    # Parse JSON จาก LLM
    try:
        # ลบ markdown code block ถ้ามี
        clean = result.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
        action = parsed.get("action", "ALLOW").upper()
        reason = parsed.get("reason", "")
    except (json.JSONDecodeError, AttributeError):
        # ถ้า parse ไม่ได้ให้ ALLOW ไว้ก่อน (fail-open) เพื่อไม่บล็อกผิด
        action = "ALLOW"
        reason = "parse error - defaulting to allow"

    if action == "BLOCK":
        print(f"    🚫 BLOCKED: {reason}")
        state["steps"].append("guardrail_input_blocked")

        # inject ข้อความปฏิเสธกลับเข้า messages โดยตรง
        # แล้ว route ไปยัง END เลย (ดูที่ graph routing ด้านล่าง)
        from langchain_core.messages import AIMessage
        rejection_msg = AIMessage(
            content=(
                "ขออภัยครับ ผมช่วยได้เฉพาะเรื่องสุขภาพและการแปลผลตรวจสุขภาพเท่านั้น\n"
                "หากมีคำถามเกี่ยวกับอาการ ผลแลป หรือโรคที่เกี่ยวข้อง ยินดีช่วยเสมอครับ"
            )
        )
        return {
            "messages": [rejection_msg],
            "blocked": True,
            "current_node": "guardrail_input",
            "steps": state["steps"] + ["guardrail_input_blocked"]
        }

    print(f"    ✅ ALLOWED: {reason}")
    return {
        "blocked": False,
        "current_node": "guardrail_input",
        "steps": state.get("steps", []) + ["guardrail_input_allowed"]
    }


def route_after_input_guardrail(state: AgentState):
    """Route หลัง input guardrail: ถ้าถูก block ให้จบเลย"""
    if state.get("blocked", False):
        return "blocked"
    return "continue"

def guardrail_output_node(state: AgentState):
    """
    ตรวจสอบ output ก่อนส่งให้ user
    """
    state["current_node"] = "guardrail_output"
    state["steps"].append("guardrail_output")

    last_ai_message = state["messages"][-1].content

    print(f"[3] 🛡️ OUTPUT GUARDRAIL: Checking safety rules...")

    guard_prompt = (
        "คุณคือหัวหน้าพยาบาลผู้ตรวจทานข้อความ (Safety Editor)\n"
        "ตรวจสอบคำตอบของ AI ตามกฎด้านล่าง แล้วตอบเป็น JSON เท่านั้น\n\n"
        "--- กฎการตรวจสอบ ---\n"
        "1. ห้ามยืนยันว่าเป็นโรคเด็ดขาด ให้ใช้คำว่า 'มีความเสี่ยง' หรือ 'แนวโน้ม'\n"
        "2. ห้ามระบุชื่อยาหรือวิธีใช้ยา\n"
        "3. ห้ามใช้คำที่ทำให้ตกใจ: อันตราย, วิกฤต, ร้ายแรง\n\n"
        "รูปแบบคำตอบ JSON (ห้ามมีข้อความอื่น):\n"
        "ถ้าผ่านทุกกฎ: "
        '{"action": "PASSED", "revised_content": null}\n'
        "ถ้าไม่ผ่าน: "
        '{"action": "MODIFIED", "revised_content": "ข้อความที่แก้ไขแล้วทั้งหมด"}\n\n'
        f"ข้อความที่ต้องตรวจ:\n{last_ai_message}"
    )

    result = chat_model.invoke(guard_prompt).content.strip()

    try:
        clean = result.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
        action = parsed.get("action", "PASSED").upper()
        revised = parsed.get("revised_content")
    except (json.JSONDecodeError, AttributeError):
        # fail-safe: ถ้า parse ไม่ได้ ให้ผ่านไปเลย ไม่แก้ไข
        action = "PASSED"
        revised = None

    if action == "MODIFIED" and revised:
        print(f"    ⚠️ MODIFIED: Response was adjusted by guardrail")
        from langchain_core.messages import AIMessage
        followup_question = _structured_followup_question_for_slot(state.get("pending_slot"))
        if followup_question:
            revised = _append_structured_followup(str(revised), followup_question)
        new_message = AIMessage(content=revised, id=state["messages"][-1].id)
        
        # ส่งกลับไปให้ LangGraph ทำการ Overwrite ข้อความเดิมตาม ID เอง
        return {
            "messages": [new_message],
            "steps": state.get("steps", []) + ["guardrail_modified"]
        }
    else:
        print("    ✅ PASSED: Response is safe")
        return {
            "steps": state.get("steps", []) + ["guardrail_passed"]
        }

# ----- Conditional -----
def should_continue(state: AgentState):
    """
    Decide whether to continue calling the tools.
    """

    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls: # ไม่มี tool_calls จาก LLM แล้ว = ต้องการจะตอบ User แล้ว

        return "end"
    else:
        return "continue"   # LLM ขอเรียกใช้ tool อยู่

def core_identity() -> str:
    return (
        "คุณคือผู้ช่วยด้านสุขภาพเบื้องต้น\n"
        "คุณไม่ใช่แพทย์ และไม่ทำการวินิจฉัยโรค\n"
        "คุณให้ข้อมูลในระดับเบื้องต้นเท่านั้น\n"
        "คุณตอบอย่างกระชับ สุภาพ และเป็นกลาง\n"
        "ใช้คำลงท้ายว่า 'ครับ'\n"
    )
 
def lab_prompt(context: str, summary_context: str) -> str:
    return (
        core_identity()
        + "\nบทบาทของคุณ:\n"
        "คุณคือผู้ช่วยด้านสุขภาพเบื้องต้น ครอบคลุม 2 กรณี:\n"
        "1. ตอบคำถามทั่วไปเกี่ยวกับโรค เช่น อาการ สาเหตุ การปฏิบัติตัว\n"
        "2. แปลผลค่าแลปเบื้องต้น โดยใช้ข้อมูลอ้างอิงจากคู่มือสุขภาพที่ให้มา\n\n"
        "ขอบเขตที่ตอบได้:\n"
        "- เบาหวาน\n"
        "- ความดันโลหิตสูง\n"
        "- ไขมันในเลือดสูง\n"
        "- โรคไตเรื้อรัง (CKD)\n"
        "- ภาวะที่เกี่ยวข้องกับการทำงานของตับ\n\n"
        f"{summary_context}\n"
        f"### ข้อมูลอ้างอิงจากคู่มือสุขภาพ:\n{context}\n\n"
        "หลักการตอบ:\n"
        "1. ถ้าผู้ใช้ถามทั่วไป ให้ตอบจากความรู้เบื้องต้น โดยอ้างอิงคู่มือด้านบนถ้าเกี่ยวข้อง\n"
        "2. ถ้าผู้ใช้ส่งค่าแลปมา ให้ตีความโดยเทียบกับข้อมูลอ้างอิงด้านบน และใช้คำว่า 'แนวโน้ม' หรือ 'ความเสี่ยงเบื้องต้น' เท่านั้น\n"
        "3. ถ้ามีค่าผลตรวจแล้ว ให้ตอบความหมายเบื้องต้นและสิ่งที่ควรทำก่อนเสมอ ห้ามเริ่มด้วยการถามอายุ เพศ หรืองดอาหารจนยังไม่ตอบคำถามหลัก\n"
        "4. หากคู่มือระบุเป้าหมายตามช่วงอายุ ให้ยึดตามนั้นเป็นหลักเมื่อมีข้อมูลอายุ\n"
        "5. สามารถให้คำเตือนเชิงพฤติกรรม เช่น ระวังหน้ามืดหรือล้ม\n\n"
        "6. ถ้าผู้ใช้ถามเรื่องหยุดยา ปรับยา หรือมีอาการ/ค่าที่เสี่ยง ให้ตอบ safety action ก่อนทันที "
        "ห้ามรอถามอายุ เพศ หรืองดอาหารก่อนตอบคำถามหลัก และให้ถามข้อมูลเพิ่มเฉพาะหลังจากให้คำแนะนำความปลอดภัยแล้ว\n"
        "7. ถ้าผู้ใช้ทักทายหรือถามว่าระบบช่วยเรื่องอะไรได้บ้าง และยังไม่มีผลตรวจ อายุ หรือเพศ "
        "ให้ตอบขอบเขตสั้นๆ แล้วถามต่อให้ผู้ใช้ส่งค่าผลตรวจพร้อมอายุและเพศเสมอ\n\n"
        "รูปแบบคำตอบ:\n"
        "- ตอบเป็น bullet point สั้นๆ ไม่เกิน 5 ข้อ\n"
        "- แต่ละข้อไม่เกิน 1-2 ประโยค\n"
        "- ปิดท้ายด้วยคำแนะนำเชิงพฤติกรรมถ้ามี\n\n"
        "ห้าม:\n"
        "- ให้แผนการรักษาหรือระบุยา\n"
        "- สรุปเป็นการวินิจฉัยโรค\n"
        "- ตอบยาวเกิน 10 ประโยค\n\n"
        "ถ้าข้อมูลไม่พอ:\n"
        "ให้ตอบเท่าที่บอกได้ก่อน แต่ห้ามถามอายุ เพศ หรืองดอาหารเองในคำตอบ "
        "เพราะระบบจะเติมคำถาม follow-up แบบคงที่ให้หลังคำตอบโดยอัตโนมัติ "
        "ไม่ถามข้อมูลที่ไม่เกี่ยว เช่น ไม่ถามงดอาหารในเคส eGFR/ยา/ไตที่ไม่ต้องใช้บริบทนี้\n"
    )
 
def no_context_prompt() -> str:
    return (
        core_identity()
        + "\nขออภัยครับ ข้อมูลที่ถามอยู่นอกขอบเขตที่ผมช่วยได้ในตอนนี้\n"
        "ผมช่วยแปลผลเบื้องต้นได้เฉพาะเรื่อง เบาหวาน ความดัน ไขมัน โรคไต และตับ\n"
        "หากมีผลแลปในหัวข้อเหล่านี้ รบกวนส่งค่าผลตรวจพร้อมอายุและเพศของผู้ที่เป็นเจ้าของผลตรวจครับ\n"
    )
    
def _analysis_slot_context(state: AgentState) -> str:
    labs = state.get("extracted_lab_values") or {}
    lab_text = ", ".join(f"{name}: {value}" for name, value in labs.items()) or "-"
    return (
        "\n\nข้อมูล structured ที่สกัดได้ก่อนวิเคราะห์:"
        f"\n- intent: {state.get('intent')}"
        f"\n- age: {state.get('age')}"
        f"\n- gender: {state.get('gender')}"
        f"\n- fasting_status: {state.get('fasting_status')}"
        f"\n- underlying_disease: {state.get('underlying_disease')}"
        f"\n- current_medications: {state.get('current_medications')}"
        f"\n- current_symptoms: {state.get('current_symptoms')}"
        f"\n- extracted_lab_values: {lab_text}"
    )


def _analysis_retrieval_query(state: AgentState, fallback_query: str) -> str:
    labs = state.get("extracted_lab_values") or {}
    if not labs:
        return fallback_query

    query_parts = [
        "health checkup lab interpretation",
        "diabetes dyslipidemia kidney liver blood pressure",
        f"age {state.get('age')}",
        f"gender {state.get('gender')}",
        f"fasting {state.get('fasting_status')}",
        f"intent {state.get('intent')}",
    ]
    query_parts.extend(f"{name} {value}" for name, value in labs.items())
    return " ".join(str(part) for part in query_parts if part)


def _structured_followup_question_for_slot(pending_slot: Optional[str]) -> Optional[str]:
    if pending_slot == "gender":
        return (
            "เพศของผู้ที่เป็นเจ้าของผลตรวจคือชายหรือหญิงครับ? "
            "ข้อมูลนี้ช่วยให้เทียบช่วงอ้างอิงได้เหมาะสมขึ้น"
        )

    if pending_slot == "age":
        return "ผู้ที่เป็นเจ้าของผลตรวจอายุเท่าไรครับ?"

    if pending_slot == "fasting_status":
        return "ผลเลือดชุดนี้ตรวจหลังงดอาหารหรือไม่ครับ? กรุณาตอบว่าใช่หรือไม่ใช่"

    return None


def _next_structured_followup(state: AgentState) -> Tuple[Optional[str], Optional[str]]:
    labs = state.get("extracted_lab_values") or {}
    if not labs or state.get("pending_slot"):
        return None, None

    if state.get("gender") is None:
        return "gender", _structured_followup_question_for_slot("gender")

    if state.get("age") is None:
        return "age", _structured_followup_question_for_slot("age")

    if state.get("fasting_status") is None and any(name in FASTING_FOLLOWUP_LABS for name in labs):
        return "fasting_status", _structured_followup_question_for_slot("fasting_status")

    return None, None


def _append_structured_followup(content: str, followup_question: str) -> str:
    trimmed_content = content.rstrip()
    if followup_question in trimmed_content:
        return trimmed_content
    return f"{trimmed_content}\n\n{followup_question}"


def call_model(state: AgentState):
    """
    Node สำหรับตอบคำถาม: ดึง Context มาใส่ใน Prompt จริงๆ
    """
    messages = state["messages"]
    summary = state.get("summary", "")
    last_user_message = messages[-1].content 
    
    # 1. ดึงข้อมูลจาก Vector DB (Markdown)
    retrieval_query = _analysis_retrieval_query(state, last_user_message)
    context = retrieve_context(retrieval_query)
    
    # Adding smurrized chat to the System Message
    summary_context = f"\n\nสรุปบริบทการสนทนาก่อนหน้านี้: {summary}" if summary else ""
    
    summary_context += _analysis_slot_context(state)
    
    print("=== RETRIEVAL QUERY ===", retrieval_query)
    print("=== CONTEXT ===", context)
    
    # 2. เลือก prompt ตามว่ามี context หรือไม่
    system_prompt = lab_prompt(context, summary_context) if context else no_context_prompt()

    # 3. ส่งคำสั่งที่มี "ข้อมูลอ้างอิง (Context)" ไปให้ Gemini
    print(f"[2] >>> AGENT NODE: Generating response...")
    response = chat_model.invoke([SystemMessage(content=system_prompt)] + messages)
    pending_slot, followup_question = _next_structured_followup(state)
    if pending_slot and followup_question:
        updated_content = _append_structured_followup(str(response.content), followup_question)
        if hasattr(response, "model_copy"):
            response = response.model_copy(update={"content": updated_content})
        else:
            response = response.copy(update={"content": updated_content})
    
    result = {
        "messages": [response], 
        "steps": state.get("steps", []) + ["retrieval", "generate"]
    }
    if pending_slot:
        result["pending_slot"] = pending_slot

    return result

# ----- Generate graph -----
def build_graph():
    graph = StateGraph(AgentState)

    # ----- add nodes -----
    graph.add_node("input", input_node)
    graph.add_node("guardrail_input", guardrail_input_node)
    graph.add_node("extract_info_node", extract_info_node)
    graph.add_node("ask_lab_node", ask_lab_node)
    graph.add_node("ask_fasting_node", ask_fasting_node)
    graph.add_node("ask_age_node", ask_age_node)
    graph.add_node("ask_gender_node", ask_gender_node)
    graph.add_node("our_agent", call_model)
    graph.add_node("guardrail_output", guardrail_output_node)
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
            "continue": "extract_info_node"
        }
    )

    graph.add_conditional_edges(
        "extract_info_node",
        route_after_extraction,
        {
            "ask_lab_node": "ask_lab_node",
            "ask_fasting_node": "ask_fasting_node",
            "ask_age_node": "ask_age_node",
            "ask_gender_node": "ask_gender_node",
            "our_agent": "our_agent",
        },
    )

    # Question nodes finish the current turn. The next user reply re-enters
    # this graph at "input" with the accumulated chat history/state.
    graph.add_edge("ask_lab_node", END)
    graph.add_edge("ask_fasting_node", END)
    graph.add_edge("ask_age_node", END)
    graph.add_edge("ask_gender_node", END)

    graph.add_conditional_edges(
        "our_agent",
        should_continue,
        {
            "continue": "tools",
            "end": "guardrail_output"
        }
    )

    graph.add_edge("guardrail_output", "summarize")
    graph.add_edge("summarize", END)
    graph.add_edge("tools", "our_agent")

    app = graph.compile()

    return app
