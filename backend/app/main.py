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
# Monitoring
import time
import psutil
import functools
import logging
from botocore.exceptions import ClientError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_embeddings():
    s3 = boto3.client(
        's3',
        region_name=os.getenv('AWS_REGION', 'us-east-2')  # Keep your fallback
    )
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




# EmbeddingRecommender class and other necessary components
from app.recommender import EmbeddingRecommender, AsyncOpenAIClient, CosineSimilarityCalculator

# Load environment variables from the project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: 
    # Load data
    # download_embeddings()
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
        try:
            # Logging
            logger.info("Initializing recommender...")
            logger.info(f"Memory before loading pkl: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
            
            # First verify the file exists and its size
            pkl_path = "embeddings.pkl"
            file_size = os.path.getsize(pkl_path)
            logger.info(f"Found embeddings.pkl with size: {file_size / 1024 / 1024:.2f} MB")
            
            # Load the course data from pkl
            try:
                logger.info("Reading pkl...")
                df = pd.read_pickle(pkl_path)
                logger.info(f"Successfully read pickle file with {len(df)} rows")
            except Exception as e:
                logger.error(f"Reading pkl error: {str(e)}")
                logger.error("Full error details:", exc_info=True)  # This will log the full stack trace
                raise  # Re-raise the exception to stop initialization
            
            logger.info(f"Memory after loading pkl: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
            
            try:
                # Initialize components with logging
                logger.info("Initializing OpenAI client...")
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
                logger.info("OpenAI client initialized")
                
                logger.info("Initializing similarity calculator...")
                similarity_calculator = CosineSimilarityCalculator()
                logger.info("Similarity calculator initialized")
                
                logger.info("Creating EmbeddingRecommender instance...")
                recommender = EmbeddingRecommender(openai_client, similarity_calculator)
                logger.info("Loading courses into recommender...")
                recommender.load_courses(df.to_dict('records'))
                logger.info("Successfully initialized recommender")
                
                logger.info(f"Memory after full initialization: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
            except Exception as e:
                logger.error(f"Error during recommender initialization: {str(e)}")
                logger.error("Full error details:", exc_info=True)
                raise
                
        except Exception as e:
            logger.error(f"Fatal error in get_recommender: {str(e)}")
            logger.error("Full error details:", exc_info=True)
            raise
            
    return recommender

# def get_recommender():
#     """
#     Creates and returns an instance of EmbeddingRecommender.
#     This function is used as a dependency, ensuring we only create one instance.
#     """
#     global recommender
#     if recommender is None:
#         # Logging
#         logger.info("Initializing recommender...")
#         logger.info(f"Memory before loading pkl: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
        
#         # Load the course data from pkl
#         pkl_path = "embeddings.pkl"
#         try:
#             logger.info("Reading pkl...")
#             df = pd.read_pickle(pkl_path)
#         except Exception as e:
#             logger.error(f"Reading pkl error: {str(e)}")

#         logger.info(f"Memory after loading pkl: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
        
#         # Rest of your initialization...
#         logger.info(f"Memory after full initialization: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
        
#         # Initialize the AsyncOpenAIClient
#         config = {
#             "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
#             "OPENAI_API_VERSION": os.getenv("OPENAI_API_VERSION"),
#             "OPENAI_API_BASE": os.getenv("OPENAI_API_BASE"),
#             "OPENAI_ORGANIZATION_ID": os.getenv("OPENAI_ORGANIZATION_ID"),
#             "GENERATOR_MODEL": os.getenv("GENERATOR_MODEL"),
#             "RECOMMENDER_MODEL": os.getenv("RECOMMENDER_MODEL"),
#             "OPENAI_EMBEDDING_MODEL": os.getenv("OPENAI_EMBEDDING_MODEL")
#         }
#         openai_client = AsyncOpenAIClient(config)
        
#         # Initialize the CosineSimilarityCalculator
#         similarity_calculator = CosineSimilarityCalculator()
        
#         # Initialize the EmbeddingRecommender
#         recommender = EmbeddingRecommender(openai_client, similarity_calculator)
#         recommender.load_courses(df.to_dict('records'))
#         logger.info("Successfully initialized recommender")
#     return recommender


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

# @app.get("/health")
# async def health_check():
#     """
#     Health check endpoint to verify if the service and its dependencies are functioning correctly.
#     """
#     print("IN BACKEND HEALTH CHECK")
#     try:
#         # Perform a simple operation to check if the recommender is working
#         recommender = get_recommender()
#         test_query = "test query"
#         async for _ in recommender.stream_recommend(query=test_query, levels=[]):
#             break
#         return {"status": "healthy"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Service unhealthy: {str(e)}")
    
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
        
        return {
            "embeddings_file_exists": file_exists,
            "embeddings_file_size_mb": file_size / (1024 * 1024),
            "current_memory_mb": memory_usage,
            "recommender_initialized": recommender_initialized
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)