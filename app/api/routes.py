from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.post("/fit_lut")
async def fit_lut() -> None:
    raise HTTPException(status_code=501, detail="fit_lut not yet implemented")


@router.post("/curate")
async def curate() -> None:
    raise HTTPException(status_code=501, detail="curate not yet implemented")


@router.get("/luts/curated/{lut_id}.cube")
async def get_curated_lut(lut_id: str) -> None:
    raise HTTPException(status_code=501, detail="curated LUTs not yet available")
