from typing import Annotated, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]

    # Runtime/debug metadata used by the existing graph.
    summary: str
    steps: list[str]
    current_node: Optional[str]
    blocked: bool

    # Per-turn semantic intent. It is recomputed from the latest message and is
    # never inferred from saved laboratory values.
    intent: Optional[str]

    # Slot-filling / medical checkup fields.
    health_state: Optional[Dict[str, object]]
    profile_metrics: Optional[object]
    age: Optional[int]
    gender: Optional[str]  # "male" or "female"
    underlying_disease: Optional[List[str]]
    current_medications: Optional[List[str]]
    current_symptoms: Optional[List[str]]
    fasting_status: Optional[str]  # "yes" or "no"
    extracted_lab_values: Optional[Dict[str, float]]
    # Values the user reported this turn that a catalog field recognized but
    # rejected as outside its allowed range (id -> {value, min, max}). Reset
    # every turn so the analyst never re-warns about an old, already-handled
    # rejection. It exists so the reply can honestly say a value was NOT saved
    # instead of defaulting to a polite "รับทราบ...บันทึกแล้วครับ".
    rejected_lab_values: Optional[Dict[str, Dict[str, float]]]
    pending_slot: Optional[str]
    # True only for a turn that updates non-laboratory profile measurements
    # without asking for an interpretation. These turns end with an
    # acknowledgement rather than RAG.
    profile_update_only: Optional[bool]
    profile_update_fields: Optional[List[str]]
    # A request for a dashboard-style health overview. It may summarize saved
    # values, but intentionally does not display textbook citations.
    health_overview_request: Optional[bool]

    # Source records for citations appended after the safety review.
    citations: Optional[List[Dict[str, object]]]
