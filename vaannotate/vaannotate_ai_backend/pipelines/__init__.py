from typing import TYPE_CHECKING, Any

from .active_learning import ActiveLearningPipeline
from .inference import InferencePipeline
from .prompt_tasks import (
    FamilyPromptTask,
    SinglePromptTask,
    df_to_family_prompt_tasks,
    df_to_single_prompt_tasks,
    family_prompt_tasks_to_df,
    single_prompt_tasks_to_df,
)

if TYPE_CHECKING:  # pragma: no cover
    # Import lazily at runtime to avoid circular import during orchestration setup.
    from .large_corpus_jobs import PromptInferenceJob, PromptPrecomputeJob


def __getattr__(name: str) -> Any:
    if name in {"PromptInferenceJob", "PromptPrecomputeJob"}:
        from .large_corpus_jobs import PromptInferenceJob, PromptPrecomputeJob

        return {  # type: ignore[return-value]
            "PromptInferenceJob": PromptInferenceJob,
            "PromptPrecomputeJob": PromptPrecomputeJob,
        }[name]
    raise AttributeError(name)


__all__ = [
    "ActiveLearningPipeline",
    "InferencePipeline",
    "FamilyPromptTask",
    "SinglePromptTask",
    "df_to_family_prompt_tasks",
    "df_to_single_prompt_tasks",
    "family_prompt_tasks_to_df",
    "single_prompt_tasks_to_df",
    "PromptInferenceJob",
    "PromptPrecomputeJob",
]
