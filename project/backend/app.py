from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from .logic import semantic_search, calculate_match_score
from .database import store_resume
from .rag import justify_candidate, get_rag_explanation
from .loader import load_resumes_from_disk

app = FastAPI(title="AI Resume Matching System")

# Data Models
class ResumeUpload(BaseModel):
    id: str
    text: str

class SearchQuery(BaseModel):
    query: str

class MatchQuery(BaseModel):
    job_description: str
    resume_text: str

# Startup Event: Automatically load resumes from dataset
@app.on_event("startup")
async def startup_event():
    load_resumes_from_disk()

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Resume Matching System API"}

@app.post("/upload-resume")
def upload_resume(data: ResumeUpload):
    """Step 6.1: Upload resume and store embedding."""
    success = store_resume(data.text, data.id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to store resume")
    return {"message": f"Resume {data.id} uploaded and indexed successfully"}

@app.post("/search")
def search(query: SearchQuery):
    """Step 6.2: Semantic search for top 3 resumes + RAG justification."""
    results = semantic_search(query.query)
    if not results:
        return {"results": [], "justification": "No matches found."}
    
    # Add RAG justification for the top match
    justification = justify_candidate(query.query)
    
    return {
        "results": results,
        "rag_justification": justification,
        "rag_info": get_rag_explanation()
    }

@app.post("/match")
def match(data: MatchQuery):
    """Step 6.3: Calculate match percentage between JD and Resume."""
    match_data = calculate_match_score(data.job_description, data.resume_text)
    
    # Recruiter benefit explanation
    benefit = (
        "Matching scores help recruiters quantitatively prioritize candidates based on "
        "semantic alignment, reducing manual screening time by focusing on top-tier talent."
    )
    
    return {
        **match_data,
        "recruiter_benefit": benefit
    }
