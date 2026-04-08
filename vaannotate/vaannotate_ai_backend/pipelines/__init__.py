from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .active_learning import ActiveLearningPipeline
    from .inference import InferencePipeline
    from .large_corpus_jobs import PromptInferenceJob, PromptPrecomputeJob

from .prompt_tasks import (
    FamilyPromptTask,
    SinglePromptTask,
    df_to_family_prompt_tasks,
    df_to_single_prompt_tasks,
    family_prompt_tasks_to_df,
    single_prompt_tasks_to_df,
)


def __getattr__(name: str) -> Any:
    if name in {"ActiveLearningPipeline", "InferencePipeline"}:
        from .active_learning import ActiveLearningPipeline
        from .inference import InferencePipeline

        return {
            "ActiveLearningPipeline": ActiveLearningPipeline,
            "InferencePipeline": InferencePipeline,
        }[name]
    if name in {"PromptInferenceJob", "PromptPrecomputeJob"}:
        from .large_corpus_jobs import PromptInferenceJob, PromptPrecomputeJob

        return {
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
