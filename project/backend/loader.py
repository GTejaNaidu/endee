import os
import logging
from .database import store_resume

logger = logging.getLogger(__name__)

def load_resumes_from_disk(directory: str = "project/dataset/resumes"):
    """
    Step 8: Data Loading.
    Automatically indexes resumes from the dataset folder.
    """
    if not os.path.exists(directory):
        # Fallback to local path check
        directory = os.path.join(os.getcwd(), "dataset", "resumes")
        if not os.path.exists(directory):
            logger.warning(f"Dataset directory not found: {directory}")
            return

    files_loaded = 0
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Use filename as ID for simplicity
                    resume_id = os.path.splitext(filename)[0]
                    store_resume(content, resume_id)
                    files_loaded += 1
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
                
    logger.info(f"Successfully loaded {files_loaded} resumes into the system.")
