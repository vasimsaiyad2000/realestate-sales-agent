from fastapi import APIRouter, Request

from app.scheduler.state import scheduler_status

router = APIRouter(prefix="/scheduler", tags=["scheduler"])


@router.get("/status")
async def get_scheduler_status(request: Request) -> dict[str, object]:
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        return {"running": False, "job_id": None, "next_run_time": None}
    return scheduler_status(scheduler)
g