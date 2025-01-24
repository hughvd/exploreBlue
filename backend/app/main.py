# main.py
import os
from typing import List, Optional, Callable
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager, contextmanager
from pydantic import BaseModel
import pandas as pd
from dotenv import load_dotenv
import boto3
# CPU Monitoring
import time
import psutil
import functools
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CPUMonitor:
    def __init__(self):
        self.process = psutil.Process()
    
    @contextmanager
    def measure_cpu(self, request_id: str):
        """Context manager to measure CPU usage during request processing"""
        start_time = time.time()
        start_cpu_time = self.process.cpu_times()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_cpu_time = self.process.cpu_times()
            
            # Calculate CPU usage
            elapsed_time = end_time - start_time
            cpu_user = end_cpu_time.user - start_cpu_time.user
            cpu_system = end_cpu_time.system - start_cpu_time.system
            total_cpu = cpu_user + cpu_system
            
            # Calculate CPU percentage
            cpu_percent = (total_cpu / elapsed_time) * 100
            
            # Get memory usage
            memory_info = self.process.memory_info()
            
            cpu_TDP = 45
            logger.info(
                f"Request {request_id} metrics:\n"
                f"Duration: {elapsed_time:.2f}s\n"
                f"CPU Usage: {cpu_percent:.1f}%\n"
                f"User CPU Time: {cpu_user:.2f}s\n"
                f"System CPU Time: {cpu_system:.2f}s\n"
                f"CPU Power Joules: {(cpu_percent/100)*cpu_TDP*(cpu_user+cpu_system)}\n"
                f"CPU Power Watts: {(cpu_percent/100)*cpu_TDP}\n"
                f"Memory Usage: {memory_info.rss / 1024 / 1024:.1f}MB"
            )

def monitor_cpu(func: Callable):
    """Decorator to monitor CPU usage of FastAPI endpoints"""
    cpu_monitor = CPUMonitor()
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        request_id = str(time.time())  # Simple request ID based on timestamp
        
        with cpu_monitor.measure_cpu(request_id):
            return await func(*args, **kwargs)
    
    return wrapper



def download_embeddings():
    if not os.path.exists('embeddings.pkl'):
        # Download from S3
        s3 = boto3.client('s3')
        bucket_name = 'course-recommender-embeddings'
        key = 'embeddings.pkl'  # The key (path) to your embeddings file in the S3 bucket
        s3.download_file(bucket_name, key, 'embeddings.pkl')
    else:
        print("Using local embeddings.pkl")



# EmbeddingRecommender class and other necessary components
from app.recommender import EmbeddingRecommender, AsyncOpenAIClient, CosineSimilarityCalculator

# Load environment variables from the project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: 
    # Load data
    download_embeddings()
    # Initialize the recommender
    get_recommender()
    yield
    # Shutdown: Clean up resources if needed
    # This runs when the application is shutting down
    pass

# Initialize FastAPI app
app = FastAPI(title="Course Recommender API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with frontend domain
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

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
        # NOTE: May be issue from running locally as container sets python path to ./backend
        pkl_path = "embeddings.pkl"
        df = pd.read_pickle(pkl_path)
        
        # Initialize the AsyncOpenAIClient
        config = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "OPENAI_API_VERSION": os.getenv("OPENAI_API_VERSION"),
            "OPENAI_API_BASE": os.getenv("OPENAI_API_BASE"),
            "OPENAI_ORGANIZATION_ID": os.getenv("OPENAI_ORGANIZATION_ID"),
            "GENERATOR_MODEL": os.getenv("GENERATOR_MODEL"),
            "RECOMMENDER_MODEL": os.getenv("RECOMMENDER_MODEL"),
            "OPENAI_EMBEDDING_MODEL": os.getenv("OPENAI_EMBEDDING_MODEL")
        }
        openai_client = AsyncOpenAIClient(config)
        
        # Initialize the CosineSimilarityCalculator
        similarity_calculator = CosineSimilarityCalculator()
        
        # Initialize the EmbeddingRecommender
        recommender = EmbeddingRecommender(openai_client, similarity_calculator)
        recommender.load_courses(df.to_dict('records'))
    return recommender

@app.get("/")
async def root():
    """
    Root endpoint to check if the API is running.
    """
    return {"message": "Course Recommender API is running"}

# @app.post("/recommend")
# async def recommend_courses(
#     request: RecommendationRequest,
#     recommender: EmbeddingRecommender = Depends(get_recommender)
# ):
#     """
#     Endpoint to get course recommendations based on user query and preferred levels.
#     Returns a streaming response of OpenAI tokens.

#     Args:
#     - request (RecommendationRequest): Contains the user's query and preferred course levels.
#     - recommender (EmbeddingRecommender): Instance of the recommender, injected as a dependency.

#     Returns:
#     - StreamingResponse: A stream of tokens from the OpenAI API.
#     """
#     return StreamingResponse(
#         recommender.stream_recommend(levels=request.levels, query=request.query),
#         media_type="text/plain"
#     )

@app.post("/recommend")
@monitor_cpu # Log cpu power
async def recommend_courses(
    request: RecommendationRequest,
    recommender: EmbeddingRecommender = Depends(get_recommender)
):
    """
    Endpoint to get course recommendations based on user query and preferred levels.
    Returns a string response of recommendations.

    Args:
    - request (RecommendationRequest): Contains the user's query and preferred course levels.
    - recommender (EmbeddingRecommender): Instance of the recommender, injected as a dependency.

    Returns:
    - str: A string containing the course recommendations.
    """
    recommendation = await recommender.recommend(query=request.query, levels=request.levels)
    return recommendation

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