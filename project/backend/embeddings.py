import logging
from sentence_transformers import SentenceTransformer
from typing import List, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model once (Singleton-like behavior for efficiency)
# 'all-MiniLM-L6-v2' is a fast, versatile model for general-purpose embeddings (384-dimensional)
try:
    logger.info("Loading SentenceTransformer model: all-MiniLM-L6-v2...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading embedding model: {e}")
    model = None

def get_embedding(text: str) -> List[float]:
    """
    Generate a vector embedding for a single text string.
    
    Args:
        text (str): The input text to embed.
        
    Returns:
        List[float]: The 384-d vector embedding as a list.
    """
    if not text.strip() or model is None:
        return []
        
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()

def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Generate vector embeddings for a batch of text strings.
    This is significantly faster than calling get_embedding sequentially.
    
    Args:
        texts (List[str]): A list of text strings.
        
    Returns:
        List[List[float]]: A list of embeddings.
    """
    if not texts or model is None:
        return []
        
    # filter out empty strings while keeping track of indices if necessary, 
    # but for simplicity handle them inline or assume valid input
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.tolist()

if __name__ == "__main__":
    # Test single embedding
    sample = "Software Engineer with Python experience"
    vector = get_embedding(sample)
    print(f"Single Test -> Length: {len(vector)}, Start: {vector[:3]}")
    
    # Test batch embedding
    batch = ["React Developer", "Data Scientist", "DevOps Engineer"]
    vectors = get_embeddings_batch(batch)
    print(f"Batch Test -> Count: {len(vectors)}, First Vector Length: {len(vectors[0])}")
