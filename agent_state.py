from typing import TypedDict, Optional, Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    session_id: str
    current_plan: Optional[dict]
    state_analysis: Optional[dict]
    tool_call_count: int
