from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="mimicamera-api",
    version="0.0.1",
    description="Backend for Mimicamera: fits a 3D LUT from reference photos.",
)

app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
