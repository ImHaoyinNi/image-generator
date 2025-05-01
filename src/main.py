from fastapi import FastAPI
from router.router import router

app = FastAPI(title="SDXL Image Generator API")

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
