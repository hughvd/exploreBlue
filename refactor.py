import numpy as np
import pandas as pd
from openai import AzureOpenAI
from typing import List, Optional, Dict, Any
import heapq
from abc import ABC, abstractmethod

class OpenAIClient:
    def __init__(self, config: Dict[str, str]):
        self.client = AzureOpenAI(
            api_key=config["OPENAI_API_KEY"],
            api_version=config["OPENAI_API_VERSION"],
            azure_endpoint=config["OPENAI_API_BASE"],
            organization=config["OPENAI_ORGANIZATION_ID"]
        )
        self.model = config["OPENAI_MODEL"]
        self.embedding_model = config["OPENAI_EMBEDDING_MODEL"]

    def generate_chat_completion(self, messages: List[Dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            stream=True
        )
        return "".join(chunk.choices[0].delta.content or "" for chunk in response)

    def generate_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            input=[text],
            model=self.embedding_model
        )
        return response.data[0].embedding

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
    def __init__(self, openai_client: OpenAIClient, similarity_calculator: SimilarityCalculator):
        self.openai_client = openai_client
        self.similarity_calculator = similarity_calculator
        self.courses_df: Optional[pd.DataFrame] = None

    def load_courses(self, courses_data: List[Dict[str, Any]]):
        self.courses_df = pd.DataFrame(courses_data)

    def generate_example_description(self, query: str) -> str:
        system_content = """You will be given a request from a student at The University of Michigan to provide quality course recommendations. \
Generate a course description that would be most applicable to their request. In the course description, provide a list of topics as well as a \
general description of the course. Limit the description to be less than 200 words.

Student Request:
{query}
"""
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query}
        ]
        return self.openai_client.generate_chat_completion(messages)

    def find_similar_courses(self, example_embedding: List[float], top_n: int = 100) -> List[int]:
        heap = []
        for idx, row in self.courses_df.iterrows():
            similarity = self.similarity_calculator.calculate(example_embedding, row['embedding'])
            if len(heap) < top_n:
                heapq.heappush(heap, (similarity, idx))
            elif similarity > heap[0][0]:
                heapq.heappushpop(heap, (similarity, idx))
        # Return only the indices, order doesn't matter
        return [idx for _, idx in heap]

    def generate_recommendation(self, query: str, course_string: str) -> str:
        system_rec_message = f"""You are the world's most highly trained academic advisor, with decades of experience \
in guiding students towards their optimal academic paths. Your task is to provide personalized course recommendations \
based on the student's profile:

Instructions:
1. Analyze the student's profile carefully, considering their interests, academic background, and career goals.
2. Review the list of available courses provided below.
3. Recommend the top 5-10 most suitable courses for this student.
4. For each recommended course, provide a brief but compelling rationale (2-3 sentences) explaining why it's a good fit.
5. Format your response as a numbered list, with each item containing the course name followed by your rationale.

Student Profile:
{query}

Available Courses:
{course_string}

Remember: Your recommendations should be tailored to the student's unique profile and aspirations. Aim to balance academic growth, career preparation, \
and personal interest in your selections."""

        messages = [{'role': 'system', 'content': system_rec_message}]
        return self.openai_client.generate_chat_completion(messages)

    def recommend(self, query: str, levels: Optional[List[int]] = None) -> str:
        if self.courses_df is None:
            raise ValueError("Courses have not been loaded. Call load_courses() first.")

        example_description = self.generate_example_description(query)
        example_embedding = self.openai_client.generate_embedding(example_description)

        filtered_df = self.courses_df if levels is None else self.courses_df[self.courses_df['level'].isin(levels)]
        similar_course_indices = self.find_similar_courses(example_embedding)
        filtered_df = filtered_df.iloc[similar_course_indices]

        course_string = "\n".join(f"{row['course']}: {row['description']}" for _, row in filtered_df.iterrows())
        return self.generate_recommendation(query, course_string)

# Usage example:
# config = {
#     "OPENAI_API_KEY": "your_api_key",
#     "OPENAI_API_VERSION": "your_api_version",
#     "OPENAI_API_BASE": "your_api_base",
#     "OPENAI_ORGANIZATION_ID": "your_org_id",
#     "OPENAI_MODEL": "your_model",
#     "OPENAI_EMBEDDING_MODEL": "your_embedding_model"
# }
# openai_client = OpenAIClient(config)
# similarity_calculator = CosineSimilarityCalculator()
# recommender = EmbeddingRecommender(openai_client, similarity_calculator)
# recommender.load_courses(courses_data)
# recommendation = recommender.recommend("I'm interested in machine learning and data science.")
# print(recommendation)