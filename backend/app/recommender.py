import numpy as np
import pandas as pd
from openai import AsyncAzureOpenAI
from typing import List, Optional, Dict, Any, AsyncGenerator, Set
import heapq
from abc import ABC, abstractmethod
import logging
import time
import asyncio
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define API request rate limiting
MAX_CONCURRENT_API_CALLS = 5
api_semaphore = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)

class AsyncOpenAIClient:
    def __init__(self, config: Dict[str, str]):
        self.client = AsyncAzureOpenAI(
            api_key=config["OPENAI_API_KEY"],
            api_version=config["OPENAI_API_VERSION"],
            azure_endpoint=config["OPENAI_API_BASE"],
            organization=config["OPENAI_ORGANIZATION_ID"]
        )
        self.generator_model = config["GENERATOR_MODEL"]
        self.rec_model = config["RECOMMENDER_MODEL"]
        self.embedding_model = config["OPENAI_EMBEDDING_MODEL"]
        
        # Add caching for embeddings to reduce API calls
        self.embedding_cache = {}

    async def generate_chat_completion(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        try:
            async with api_semaphore:
                response = await self.client.chat.completions.create(
                    model=self.rec_model,
                    messages=messages,
                    temperature=0,
                    stream=True
                )
                async for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Error in generate_chat_completion: {str(e)}")
            raise

    async def generate_embedding(self, text: str) -> List[float]:
        # Check cache first
        if text in self.embedding_cache:
            logger.info("Using cached embedding")
            return self.embedding_cache[text]
        
        try:
            async with api_semaphore:
                response = await self.client.embeddings.create(
                    input=[text],
                    model=self.embedding_model
                )
                embedding = response.data[0].embedding
                # Cache the result
                self.embedding_cache[text] = embedding
                return embedding
        except Exception as e:
            logger.error(f"Error in generate_embedding: {str(e)}")
            raise

class SimilarityCalculator(ABC):
    @abstractmethod
    def calculate(self, vec1: List[float], vec2: List[float]) -> float:
        pass

class CosineSimilarityCalculator(SimilarityCalculator):
    def calculate(self, vec1: List[float], vec2: List[float]) -> float:
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

class EmbeddingRecommender:
    def __init__(self, openai_client: AsyncOpenAIClient, similarity_calculator: SimilarityCalculator):
        self.openai_client = openai_client
        self.similarity_calculator = similarity_calculator
        self.courses_by_level = {}
        
        # Add concurrency control
        self.embedding_lock = asyncio.Lock()

    def load_courses(self, courses_data: List[Dict[str, Any]]):
        """Load courses data and pre-organize by level"""
        temp = pd.DataFrame(courses_data)
        
        # Pre-compute level filtering for common course levels
        for level in range(100, 1000, 100):
            self.courses_by_level[level] = temp[temp['level'] == level]
    
    async def get_filtered_courses(self, levels: Optional[List[int]] = None) -> pd.DataFrame:
        """Get courses filtered by level with caching for repeated queries"""
        if not self.courses_by_level:
            raise ValueError("Courses have not been loaded. Call load_courses() first.")
        # Apply filtering
        if levels is None:
            filtered_df = pd.concat(list(self.courses_by_level.values()))
        else:
            filtered_df = pd.concat([self.courses_by_level[level] for level in levels])

        return filtered_df

    async def generate_example_description(self, query: str) -> str:
        """Generate an example description based on the query"""
        system_content = f"""You will be given a request from a student at The University of Michigan to provide quality course recommendations. \
Generate a course description that would be most applicable to their request. In the course description, provide a list of topics as well as a \
general description of the course. Limit the description to be less than 200 words.

Student Request:
{query}
"""
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query}
        ]
        try:
            async with api_semaphore:
                response = await self.openai_client.client.chat.completions.create(
                    model=self.openai_client.generator_model,
                    messages=messages,
                    temperature=0,
                    stream=False
                )
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating example description: {str(e)}")
            raise

    def find_similar_courses(self, filtered_df: pd.DataFrame, example_embedding: List[float], top_n: int = 50) -> List[int]:
        """Find most similar courses based on embedding similarity"""
        # Use a more efficient approach for large dataframes
        if len(filtered_df) > 1000:
            # Convert to numpy arrays for vectorized operations
            embeddings = np.array(filtered_df['embedding'].tolist())
            example_np = np.array(example_embedding)
            
            # Compute all similarities at once (vectorized)
            dot_products = np.dot(embeddings, example_np)
            norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(example_np)
            similarities = dot_products / norms
            
            # Get indices of top_n highest similarities
            top_indices = np.argsort(similarities)[-top_n:][::-1]
            return filtered_df.index[top_indices].tolist()
        else:
            # Original implementation for smaller dataframes
            heap = []
            for idx, row in filtered_df.iterrows():
                similarity = self.similarity_calculator.calculate(example_embedding, row['embedding'])
                if len(heap) < top_n:
                    heapq.heappush(heap, (similarity, idx))
                elif similarity > heap[0][0]:
                    heapq.heappushpop(heap, (similarity, idx))
            return [idx for _, idx in heap]

    async def stream_recommend(self, query: str, levels: Optional[List[int]] = None) -> AsyncGenerator[str, None]:
        """Generate streaming recommendation response"""
        try:
            if not self.courses_by_level:
                raise ValueError("Courses have not been loaded. Call load_courses() first.")

            # Generate example description and embedding concurrently
            example_description = await self.generate_example_description(query)
            
            # Get embedding with appropriate concurrency control
            async with self.embedding_lock:
                example_embedding = await self.openai_client.generate_embedding(example_description)

            # Filter courses and find similar ones
            filtered_df = await self.get_filtered_courses(levels)
            filtered_df = filtered_df.reset_index(drop=True)
            
            similar_course_indices = self.find_similar_courses(filtered_df, example_embedding)
            filtered_df = filtered_df.iloc[similar_course_indices]

            # Prepare course string for the prompt
            course_string = "\n".join(f"{row['course']}: {row['title']}\n{row['description']}" for _, row in filtered_df.iterrows())

            # Prepare the recommendation prompt
            system_rec_message = f"""You are an expert academic advisor specializing in personalized course recommendations. \
When evaluating matches between student profiles and courses, prioritize direct relevance and career trajectory fit.

Context: Student Profile ({query})
Course Options: 
{course_string}

REQUIREMENTS:
- Return exactly 10 courses, ranked by relevance and fit
- Recommend ONLY courses listed in Course Options
- For each recommendation include:
  1. Course number
  2. Course name
  2. Two-sentence explanation focused on student's specific profile/goals
  3. Confidence level (High/Medium/Low)

FORMAT (Markdown):
1. **COURSEXXX: COURSE_TITLE**
Rationale: [Two clear sentences explaining fit]
Confidence: [Level]

2. [Next course...]

CONSTRAINTS:
- NO general academic advice
- NO mentions of prerequisites unless explicitly stated in course description
- NO suggestions outside provided course list
- NO mention of being an AI or advisor
- **If multiple courses have identical titles and descriptions (cross-listed), recommend only ONE of them**"""

            messages = [{'role': 'system', 'content': system_rec_message}]

            # Stream the recommendation with appropriate concurrency control
            try:
                async for token in self.openai_client.generate_chat_completion(messages):
                    yield token
            except Exception as e:
                yield f"Error generating recommendation: {str(e)}"

        except Exception as e:
            yield f"Unexpected error: {str(e)}"

    async def recommend(self, query: str, levels: Optional[List[int]] = None) -> str:
        """Non-streaming recommendation function"""
        try:
            if not self.courses_by_level:
                raise ValueError("Courses have not been loaded. Call load_courses() first.")

            # Generate example description and get embedding with concurrency control
            start_time = time.time()
            example_description = await self.generate_example_description(query)
            logger.info(f"Generated example description in {time.time() - start_time:.2f}s")
            
            start_time = time.time()
            async with self.embedding_lock:
                example_embedding = await self.openai_client.generate_embedding(example_description)
            logger.info(f"Generated embedding in {time.time() - start_time:.2f}s")

            # Filter and find similar courses
            start_time = time.time()
            filtered_df = await self.get_filtered_courses(levels)
            filtered_df = filtered_df.reset_index(drop=True)
            
            similar_course_indices = self.find_similar_courses(filtered_df, example_embedding)
            filtered_df = filtered_df.iloc[similar_course_indices]
            logger.info(f"Found similar courses in {time.time() - start_time:.2f}s")

            # Prepare course string for the prompt
            course_string = "\n".join(f"{row['course']}: {row['title']}\n{row['description']}" for _, row in filtered_df.iterrows())
            
            # Prepare the recommendation prompt
            system_rec_message = f"""You are an expert academic advisor specializing in personalized course recommendations. \
When evaluating matches between student profiles and courses, prioritize direct relevance and career trajectory fit.

Context: Student Profile ({query})
Course Options: 
{course_string}

REQUIREMENTS:
- Return exactly 10 courses, ranked by relevance and fit
- Recommend ONLY courses listed in Course Options
- If a course is cross-listed, write the course number as "COURSEXXX (Cross-listed as COURSEYYY)"
- For each recommendation include:
  1. Course number (include cross-listed courses)
  2. Course name
  2. Two-sentence explanation focused on student's specific profile/goals
  3. Confidence level (High/Medium/Low)

FORMAT (Markdown):
1. **COURSEXXX: COURSE_TITLE**
Rationale: [Two clear sentences explaining fit]
Confidence: [Level]

2. [Next course...]

CONSTRAINTS:
- NO general academic advice
- NO mentions of prerequisites unless explicitly stated in course description
- NO suggestions outside provided course list
- NO mention of being an AI or advisor"""

            messages = [{'role': 'system', 'content': system_rec_message}]

            # Generate recommendation with concurrency control
            start_time = time.time()
            async with api_semaphore:
                response = await self.openai_client.client.chat.completions.create(
                    model=self.openai_client.rec_model,
                    messages=messages,
                    temperature=0,
                    max_tokens=1500,
                    stream=False
                )
            logger.info(f"Generated recommendation in {time.time() - start_time:.2f}s")
            
            recommendation = response.choices[0].message.content
            return recommendation
        except Exception as e:
            logger.error(f"Error in recommend: {str(e)}")
            return f"Error: {str(e)}"