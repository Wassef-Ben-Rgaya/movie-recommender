
from fastapi import FastAPI

from src.api.routes.recommendations import router as recommendations_router

app = FastAPI(
    title="Movie Recommender API",
    description="Personalized movie recommendations using collaborative filtering.",
    version="1.0.0",
)


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy"}


@app.get("/", tags=["Health"])
def root():
    return {
        "name": "Movie Recommender API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "recommendations": "/recommendations/{user_id}",
    }

app.include_router(recommendations_router)
