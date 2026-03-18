import numpy as np
from typing import List, Dict, Any
import re
from .embeddings import get_embedding
from .database import search_resumes

def calibrate_score(raw_score: float) -> float:
    """
    Transforms raw similarity scores into recruiter-friendly percentages (0-100%).
    
    Mapping strategy:
    - Below 0.1: Noise (0-30%)
    - 0.1 to 0.2: Low alignment (30-60%)
    - 0.2 to 0.3: Good candidate (60-80%)
    - 0.3 to 0.4: Strong match (80-90%)
    - Above 0.4: Top tier (90-100%)
    """
    if raw_score <= 0: return 0.0
    
    if raw_score < 0.1: 
        score = raw_score * 300
    elif raw_score < 0.2: 
        score = 30 + (raw_score - 0.1) * 300
    elif raw_score < 0.3: 
        score = 60 + (raw_score - 0.2) * 200
    elif raw_score < 0.4: 
        score = 80 + (raw_score - 0.3) * 100
    else:
        score = 90 + (raw_score - 0.4) * 50
    
    return min(100.0, round(score, 1))

def semantic_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Performs semantic search and returns cleaned, scored, and formatted results.
    """
    query_vector = get_embedding(query)
    if not query_vector:
        return []
    
    # Retrieve from Endee
    raw_results = search_resumes(query_vector, top_k=top_k)
    
    processed_results = []
    for res in raw_results:
        # Calibrate the math into a percentage
        match_pct = calibrate_score(res["score"])
        
        # Clean up text for preview (first 150 chars)
        preview = res["text"].replace("\n", " ").strip()
        
        # Highlight keywords
        from .rag import extract_keywords
        keywords = extract_keywords(query)
        for kw in keywords:
            # Escape keyword for regex
            kw_esc = re.escape(kw)
            # Use a boundary matching that works with punctuation (e.g. C++, .NET)
            # We want to match `kw` if it's not surrounded by word characters
            pattern = re.compile(rf'(?<![a-zA-Z0-9]){kw_esc}(?![a-zA-Z0-9])', re.IGNORECASE)
            # Use a lambda to replace with the original matched text, preserving case
            preview = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", preview)
            
        preview = preview[:170] + "..." # slightly longer to account for mark tags
        
        processed_results.append({
            "candidate_name": res["id"],
            "match_percentage": f"{match_pct}%",
            "text_preview": preview,
            "full_text": res["text"],
            "raw_score": round(res["score"], 4) # Keeping for debug/transparency
        })
        
    # Final sort just in case Endee didn't return perfectly ordered results
    return sorted(processed_results, key=lambda x: float(x["match_percentage"].replace("%", "")), reverse=True)

def calculate_match_score(job_description: str, resume_text: str) -> Dict[str, Any]:
    """
    Calculates detailed match metrics between a JD and a single Resume.
    """
    jd_vector = np.array(get_embedding(job_description))
    resume_vector = np.array(get_embedding(resume_text))
    
    if jd_vector.size == 0 or resume_vector.size == 0:
        return {"match_percentage": "0.0%", "raw_score": 0.0}
        
    # Cosine Similarity via Dot Product (vectors are normalized)
    similarity = float(np.dot(jd_vector, resume_vector))
    match_pct = calibrate_score(similarity)
    
    return {
        "match_percentage": f"{match_pct}%",
        "raw_score": round(similarity, 4)
    }

if __name__ == "__main__":
    # Internal validation
    test_jd = "Senior React Developer with cloud experience"
    test_resume = "Expert frontend engineer with 5 years of React, AWS, and TypeScript."
    
    result = calculate_match_score(test_jd, test_resume)
    print(f"Match Result: {result['match_percentage']} (Raw: {result['raw_score']})")
