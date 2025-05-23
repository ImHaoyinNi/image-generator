import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from router.router import router
from src.service.image_generator import image_generator


def task_worker():
    while True:
        try:
            image_generator.process_task()
        except Exception as e:
            print(f"Error in background task: {e}")
        time.sleep(2)

@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=task_worker, daemon=True)
    thread.start()
    yield
app = FastAPI(title="SDXL Image Generator API", lifespan=lifespan)

app.include_router(router)
@app.get("/")
async def read_root():
    return {"Hello": "World"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
