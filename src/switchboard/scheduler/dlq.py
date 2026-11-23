"""Dead-letter queue helpers (inspect / requeue). Storage lives in the tasks table (status='dead')."""
from __future__ import annotations

from .queue import TaskQueue


def inspect(queue: TaskQueue, job_id: str | None = None, limit: int = 100) -> list[dict]:
    return queue.dead_tasks(job_id, limit)


def requeue(queue: TaskQueue, job_id: str | None = None, task_ids: list[int] | None = None) -> int:
    return queue.requeue_dead(job_id, task_ids)
