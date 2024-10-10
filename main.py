# main.py

import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd
from dotenv import load_dotenv

# Import your modified EmbeddingRecommender class
from recommender import EmbeddingRecommender

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Course Recommender API")

# Pydantic models for request
class RecommendationRequest(BaseModel):
    query: str
    levels: Optional[List[int]] = None

# Global variable to store the EmbeddingRecommender instance
recommender = None

def get_recommender():
    """
    Creates and returns an instance of EmbeddingRecommender.
    This function is used as a dependency, ensuring we only create one instance.
    """
    global recommender
    if recommender is None:
        # Load the course data from pkl
        pkl_path = os.getenv("COURSE_PKL_PATH", "embeddings.pkl")
        df = pd.read_pickle(pkl_path)
        
        # Initialize the EmbeddingRecommender
        recommender = EmbeddingRecommender(df)
    return recommender

@app.on_event("startup")
async def startup_event():
    """
    Function that runs when the server starts up.
    It initializes the recommender to avoid cold start issues.
    """
    get_recommender()

@app.get("/")
async def root():
    """
    Root endpoint to check if the API is running.
    """
    return {"message": "Welcome to the Course Recommender API"}

@app.post("/recommend")
async def recommend_courses(
    request: RecommendationRequest,
    recommender: EmbeddingRecommender = Depends(get_recommender)
):
    """
    Endpoint to get course recommendations based on user query and preferred levels.
    Returns a streaming response of OpenAI tokens.

    Args:
    - request (RecommendationRequest): Contains the user's query and preferred course levels.
    - recommender (EmbeddingRecommender): Instance of the recommender, injected as a dependency.

    Returns:
    - StreamingResponse: A stream of tokens from the OpenAI API.
    """
    return StreamingResponse(
        recommender.stream_recommend(levels=request.levels, query=request.query),
        media_type="text/plain"
    )

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify if the service and its dependencies are functioning correctly.
    """
    try:
        # Perform a simple operation to check if the recommender is working
        recommender = get_recommender()
        test_query = "test query"
        async for _ in recommender.stream_recommend(query=test_query, levels=[]):
            break
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Service unhealthy: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)