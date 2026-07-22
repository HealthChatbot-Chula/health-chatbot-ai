from fastapi import APIRouter
from fastapi.responses import HTMLResponse, StreamingResponse

from backend.eval_ui import simulator_page
from services.eval_service import eval_simulation_event_stream, load_simulation_cases


router = APIRouter()


@router.get("/eval/simulator")
async def eval_simulator_page():
    return HTMLResponse(
        simulator_page(),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get("/eval/simulator/cases")
async def eval_simulator_cases():
    return [
        {
            "id": case["id"],
            "risk_level": case.get("risk_level", "-"),
            "starting_prompt": case.get("starting_prompt", ""),
        }
        for case in load_simulation_cases()
    ]


@router.get("/eval/simulator/stream/{case_id}")
async def eval_simulator_stream(case_id: str):
    return StreamingResponse(
        eval_simulation_event_stream(case_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
