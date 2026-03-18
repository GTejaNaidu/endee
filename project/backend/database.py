"""
database.py — Endee Vector Database client with in-memory fallback.

All vector storage and similarity search operations go through this module.
If Endee is unreachable at startup, the system automatically switches to a
NumPy-based in-memory cosine similarity store.

Environment variables:
    NDD_URL          Endee REST API base URL (default: http://localhost:8080/api/v1)
    NDD_AUTH_TOKEN   Optional bearer token for Endee authentication
"""
import logging
import os
import requests
import msgpack
from typing import List, Dict, Any
from .embeddings import get_embedding

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabase:
    """
    Direct integration with the Endee Vector Database using REST API.
    Fallback to in-memory store if Endee is unreachable.
    """
    def __init__(self):
        self.base_url = os.getenv("NDD_URL", "http://localhost:8080/api/v1")
        self.auth_token = os.getenv("NDD_AUTH_TOKEN", "")
        self.collection_name = "resumes"
        self.dimension = 384
        
        # Fallback storage: Active only if Endee connection fails
        self.use_fallback = False
        self.memory_store = [] # List of dicts: {"id": str, "vector": list, "text": str}
        
        self.headers = {
            "Content-Type": "application/json"
        }
        if self.auth_token:
            self.headers["Authorization"] = self.auth_token

        self.init_db()

    def init_db(self):
        """Checks if the index exists, creates it if not."""
        try:
            # 1. Attempt to connect to Endee
            list_url = f"{self.base_url}/index/list"
            response = requests.get(list_url, headers=self.headers, timeout=2)
            
            if response.status_code == 200:
                self.use_fallback = False
                logger.info("✅ SUCCESS: Connected to Endee Vector Database.")
                indexes = response.json().get("indexes", [])
                if any(idx["name"] == self.collection_name for idx in indexes):
                    logger.info(f"Index '{self.collection_name}' already exists.")
                    return

                # 2. Create index if it doesn't exist
                logger.info(f"Creating index '{self.collection_name}'...")
                create_url = f"{self.base_url}/index/create"
                data = {
                    "index_name": self.collection_name,
                    "dim": self.dimension,
                    "space_type": "cosine"
                }
                requests.post(create_url, json=data, headers=self.headers)
            else:
                raise Exception(f"Endee returned status {response.status_code}")

        except Exception as e:
            self.use_fallback = True
            logger.warning(f"⚠️ Endee DB unreachable ({e}). Using LOCAL IN-MEMORY FALLBACK.")

    def store_resume(self, text: str, resume_id: str) -> bool:
        """Generates an embedding and stores it in Endee or Fallback."""
        try:
            embedding = get_embedding(text)
            if not embedding:
                return False

            if self.use_fallback:
                # Local storage: Check if already exists to prevent duplicates
                for i, item in enumerate(self.memory_store):
                    if item["id"] == resume_id:
                        self.memory_store[i] = {
                            "id": resume_id,
                            "vector": embedding,
                            "text": text
                        }
                        logger.debug(f"[Fallback] Updated resume {resume_id} locally.")
                        return True

                # If not found, append as new
                self.memory_store.append({
                    "id": resume_id,
                    "vector": embedding,
                    "text": text
                })
                logger.debug(f"[Fallback] Stored new resume {resume_id} locally.")
                return True

            # Endee storage
            insert_url = f"{self.base_url}/index/{self.collection_name}/vector/insert"
            data = {
                "id": resume_id,
                "vector": embedding,
                "meta": text
            }
            response = requests.post(insert_url, json=data, headers=self.headers)
            return response.status_code == 200

        except Exception as e:
            logger.error(f"Error storing resume {resume_id}: {e}")
            return False

    def get_relevant_resumes(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieves top_k similar resumes from Endee or Fallback."""
        try:
            if self.use_fallback:
                if not self.memory_store:
                    return []
                
                # Perform manual cosine similarity using NumPy
                query_np = np.array(query_vector)
                results = []
                for item in self.memory_store:
                    vec_np = np.array(item["vector"])
                    norm_q = np.linalg.norm(query_np)
                    norm_v = np.linalg.norm(vec_np)
                    if norm_q == 0 or norm_v == 0:
                        score = 0.0
                    else:
                        score = float(np.dot(query_np, vec_np) / (norm_q * norm_v))
                    
                    results.append({
                        "id": item["id"],
                        "text": item["text"],
                        "score": score
                    })
                
                # Sort by score (desc) and return top_k
                results.sort(key=lambda x: x["score"], reverse=True)
                return results[:top_k]

            # Regular Endee search
            search_url = f"{self.base_url}/index/{self.collection_name}/search"
            data = {"vector": query_vector, "k": top_k}
            response = requests.post(search_url, json=data, headers=self.headers)
            
            if response.status_code == 200:
                raw_data = response.content
                results = msgpack.unpackb(raw_data, raw=False)
                return [{
                    "id": res["id"],
                    "text": res["meta"].decode("utf-8") if isinstance(res["meta"], bytes) else res["meta"],
                    "score": res["score"]
                } for res in results]
            return []

        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []

# Singleton instance
db = VectorDatabase()

def store_resume(text: str, resume_id: str) -> bool:
    return db.store_resume(text, resume_id)

def search_resumes(query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    return db.get_relevant_resumes(query_vector, top_k)
