from celery import chain

from backend.api.tasks import (
    cleanup_trace_task,
    extract_problematic_accounts,
    stage_a_task,
)


def run_full_pipeline(sid: str):
    """Run the full "Stage-A → Cleanup → Problematic" pipeline for ``sid``.

    The tasks are chained in the above order using immutable signatures so each
    step receives ``sid`` directly rather than the previous task's result.
    """
    return chain(
        stage_a_task.si(sid),
        cleanup_trace_task.si(sid),
        extract_problematic_accounts.si(sid),
    ).apply_async()
