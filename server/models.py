from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal

class PrivacyAction(BaseModel):
    """Action model for the DPO Agent."""
    operation: Literal["redact", "delete", "export", "retain", "notify"] = Field(
        ..., description="The privacy operation to perform."
    )
    target: str = Field(..., description="The field name, record ID, or user ID to operate on.")
    legal_basis: Optional[str] = Field(
        None, description="The legal justification (e.g., 'GDPR Art. 6', 'Financial retention law')."
    )
    reasoning: str = Field(..., description="Brief thought process for the action.")

class PrivacyObservation(BaseModel):
    """Observation model for the DPO Agent."""
    task_id: str
    instruction: str
    data_buffer: str = Field(..., description="The current view of the data being audited.")
    policy_context: str = Field(..., description="Company privacy policy and regional laws.")
    region: str = Field(..., description="The data storage region (e.g., 'EU-West-1', 'US-East-1').")
    step_count: int
    max_steps: int

class PrivacyReward(BaseModel):
    """Reward model scoring the compliance action."""
    value: float = Field(..., description="The reward value assigned by the grader.")
    partial_score: float = Field(..., description="Value before normalization.")
    logic_explanation: str = Field(..., description="Why this reward was given.")
    done: bool = False
