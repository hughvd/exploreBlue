# main.py
import os
from typing import List, Optional, Callable
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
import pandas as pd
from dotenv import load_dotenv
import boto3
import time
import psutil
import logging
import asyncio
from botocore.exceptions import ClientError
from collections import deque

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Use a semaphore to limit concurrent API calls
MAX_CONCURRENT_RECOMMENDATIONS = 10
recommendation_semaphore = asyncio.Semaphore(MAX_CONCURRENT_RECOMMENDATIONS)

async def download_embeddings_async():
    """Asynchronous wrapper for S3 download to prevent blocking the event loop"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, download_embeddings)

def download_embeddings():
    s3 = boto3.client(
        's3',
        region_name=os.getenv('AWS_REGION', 'us-east-2')
    )
    if not os.path.exists('embeddings.pkl'):
        try:
            logger.info("Attempting to download embeddings from S3...")
            s3.download_file('course-recommender-embeddings', 'embeddings.pkl', 'embeddings.pkl')
            logger.info("Successfully downloaded embeddings file from S3")
            
        except ClientError as e:
            logger.error(f"AWS API Error: {e.response['Error']['Message']}")
            logger.debug("Full error details:", exc_info=True)  # For debugging
            raise
            
        except Exception as e:
            logger.error(f"Unexpected download error: {str(e)}")
            logger.exception("Stack trace:")  # Automatically includes traceback
            raise

# Import necessary classes
from app.recommender import EmbeddingRecommender, AsyncOpenAIClient, CosineSimilarityCalculator

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Download embeddings and initialize recommender
    await download_embeddings_async()
    # Initialize recommender (shared across all workers)
    await get_recommender() 
    yield
    # Shutdown: No cleanup needed
    pass

# Rate limiter configuration
MAX_REQUESTS_PER_PERIOD = 1  # Maximum number of requests
RATE_LIMIT_PERIOD = 60  # Time period in seconds

class RateLimiter:
    def __init__(self, max_requests: int, period: float):
        """
        Initialize the rate limiter.

        :param max_requests: Maximum number of requests allowed in the given period.
        :param period: Time period (in seconds) for the rate limit.
        """
        self.max_requests = max_requests
        self.period = period
        self.request_timestamps = deque()

    def is_request_allowed(self) -> bool:
        """
        Check if a request is allowed under the rate limit.

        :return: True if the request is allowed, False otherwise.
        """
        current_time = time.time()

        # Remove timestamps outside the current period
        while self.request_timestamps and self.request_timestamps[0] < current_time - self.period:
            self.request_timestamps.popleft()

        if len(self.request_timestamps) < self.max_requests:
            # Allow the request and record the timestamp
            self.request_timestamps.append(current_time)
            return True
        else:
            # Deny the request
            return False

# Initialize the rate limiter
rate_limiter = RateLimiter(MAX_REQUESTS_PER_PERIOD, RATE_LIMIT_PERIOD)

# Initialize FastAPI app
app = FastAPI(title="Course Recommender API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class RecommendationRequest(BaseModel):
    query: str
    levels: Optional[List[int]] = None

# Global variables - will be shared across requests in the same worker
recommender = None

async def get_recommender():
    """
    Creates and returns an instance of EmbeddingRecommender.
    This is an async function to avoid blocking during initialization.
    """
    global recommender
    if recommender is None:
        try:
            logger.info("Initializing recommender...")
            logger.info(f"Memory before loading pkl: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
            
            # Load dataframe asynchronously
            pkl_path = "embeddings.pkl"
            file_size = os.path.getsize(pkl_path)
            logger.info(f"Found embeddings.pkl with size: {file_size / 1024 / 1024:.2f} MB")

            # Load the course data from pkl
            try:
                logger.info("Reading pkl...")
                df_temp = pd.read_pickle(pkl_path)
                logger.info(f"Successfully read pickle file with {len(df_temp)} rows")
            except Exception as e:
                logger.error(f"Reading pkl error: {str(e)}")
                logger.error("Full error details:", exc_info=True)  # This will log the full stack trace
                raise  # Re-raise the exception to stop initialization
                
            logger.info(f"Memory after loading pkl: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
            
            # Initialize components
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
            similarity_calculator = CosineSimilarityCalculator()
            
            recommender = EmbeddingRecommender(openai_client, similarity_calculator)
            # Load courses from cached dataframe
            recommender.load_courses(df_temp.to_dict('records'))
            del df_temp
            logger.info("Successfully initialized recommender")
            logger.info(f"Memory after initialization: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
            
        except Exception as e:
            logger.error(f"Error during recommender initialization: {str(e)}")
            logger.error("Full error details:", exc_info=True)
            raise
            
    return recommender

@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "Course Recommender API is running"}

@app.post("/recommend")
async def recommend_courses(
    request: RecommendationRequest,
    recommender_instance: EmbeddingRecommender = Depends(get_recommender)
):
    """
    Endpoint to get course recommendations based on user query and preferred levels.
    Uses a semaphore to limit concurrent API calls and a rate limiter to enforce request limits.
    """
    # Enforce rate limiting
    if not rate_limiter.is_request_allowed():
        raise HTTPException(
            status_code=429,  # Too Many Requests
            detail=f"Rate limit exceeded. Try again later."
        )
    # Limit concurrent recommendations
    async with recommendation_semaphore:
        recommendation = await recommender_instance.recommend(
            query=request.query, 
            levels=request.levels
        )
    return recommendation

@app.get("/health")
async def health_check():
    """Simple health check"""
    logger.info("Health check requested")
    return {"status": "healthy"}
    
@app.get("/debug")
async def debug_info():
    """Debug endpoint to check application state"""
    try:
        file_exists = os.path.exists('embeddings.pkl')
        file_size = os.path.getsize('embeddings.pkl') if file_exists else 0
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
        recommender_initialized = recommender is not None
        
        # Get semaphore status
        semaphore_value = recommendation_semaphore._value
        semaphore_waiters = len(recommendation_semaphore._waiters)
        
        return {
            "embeddings_file_exists": file_exists,
            "embeddings_file_size_mb": file_size / (1024 * 1024),
            "current_memory_mb": memory_usage,
            "recommender_initialized": recommender_initialized,
            "concurrent_capacity": MAX_CONCURRENT_RECOMMENDATIONS,
            "available_slots": semaphore_value,
            "waiting_requests": semaphore_waiters
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)