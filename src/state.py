"""
Pipeline state machine enforcing stage transitions.
Stages:
INIT -> INPUTS_LOADED -> DOCUMENTS_CHUNKED -> INDEX_BUILT -> RETRIEVAL_COMPLETE
-> DRAFT_ANSWERS_GENERATED -> HUMAN_REVIEW_COMPLETE -> ANSWERS_AUDITED
-> FINAL_REPORT_GENERATED -> VALIDATION_COMPLETE -> RESULTS_FINALISED
"""
from enum import Enum
from typing import List, Optional


class PipelineStage(str, Enum):
    INIT = "INIT"
    INPUTS_LOADED = "INPUTS_LOADED"
    DOCUMENTS_CHUNKED = "DOCUMENTS_CHUNKED"
    INDEX_BUILT = "INDEX_BUILT"
    RETRIEVAL_COMPLETE = "RETRIEVAL_COMPLETE"
    DRAFT_ANSWERS_GENERATED = "DRAFT_ANSWERS_GENERATED"
    HUMAN_REVIEW_COMPLETE = "HUMAN_REVIEW_COMPLETE"
    ANSWERS_AUDITED = "ANSWERS_AUDITED"
    FINAL_REPORT_GENERATED = "FINAL_REPORT_GENERATED"
    VALIDATION_COMPLETE = "VALIDATION_COMPLETE"
    RESULTS_FINALISED = "RESULTS_FINALISED"


STAGE_ORDER: List[PipelineStage] = [
    PipelineStage.INIT,
    PipelineStage.INPUTS_LOADED,
    PipelineStage.DOCUMENTS_CHUNKED,
    PipelineStage.INDEX_BUILT,
    PipelineStage.RETRIEVAL_COMPLETE,
    PipelineStage.DRAFT_ANSWERS_GENERATED,
    PipelineStage.HUMAN_REVIEW_COMPLETE,
    PipelineStage.ANSWERS_AUDITED,
    PipelineStage.FINAL_REPORT_GENERATED,
    PipelineStage.VALIDATION_COMPLETE,
    PipelineStage.RESULTS_FINALISED,
]


class PipelineStateMachine:
    """Enforces monotonic stage progression for the RAG pipeline."""

    def __init__(self, initial_stage: PipelineStage = PipelineStage.INIT):
        self.current_stage = initial_stage
        self.history: List[PipelineStage] = [initial_stage]

    def transition_to(self, target_stage: PipelineStage) -> None:
        """Transitions to the target stage, validating that it is the exact expected next stage."""
        current_idx = STAGE_ORDER.index(self.current_stage)
        target_idx = STAGE_ORDER.index(target_stage)

        if target_idx != current_idx + 1:
            raise RuntimeError(
                f"Invalid pipeline stage transition from {self.current_stage.value} to {target_stage.value}. "
                f"Expected next stage: {STAGE_ORDER[current_idx + 1].value if current_idx + 1 < len(STAGE_ORDER) else 'None (Already finalised)'}"
            )
        self.current_stage = target_stage
        self.history.append(target_stage)

    def is_at_least(self, stage: PipelineStage) -> bool:
        """Returns True if the current stage is at or past the given stage."""
        return STAGE_ORDER.index(self.current_stage) >= STAGE_ORDER.index(stage)

    def assert_stage(self, stage: PipelineStage, error_msg: Optional[str] = None) -> None:
        """Raises RuntimeError if not at the specified stage."""
        if self.current_stage != stage:
            raise RuntimeError(error_msg or f"Expected stage {stage.value}, but currently at {self.current_stage.value}")
