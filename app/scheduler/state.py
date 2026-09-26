from apscheduler.schedulers.asyncio import AsyncIOScheduler


def scheduler_status(scheduler: AsyncIOScheduler) -> dict[str, object]:
    job = scheduler.get_job("project-rag-sync")
    return {
        "running": scheduler.running,
        "job_id": job.id if job else None,
        "next_run_time": job.next_run_time.isoformat() if job and job.next_run_time else None,
    }
