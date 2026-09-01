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

    # Intent classification.
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
    pending_slot: Optional[str]
