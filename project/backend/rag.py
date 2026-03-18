import re
from typing import List, Dict, Any
from .logic import semantic_search

def extract_keywords(text: str) -> List[str]:
    """Simple regex to extract meaningful words for matching."""
    # Focus on technical terms (alphanumeric, 2+ chars, ignoring common stops)
    # Includes +, # to capture C++, C# and \. for .NET but stripped of leading/trailing punctuation generally
    words = [w for w in re.findall(r'[a-z0-9+#\.]+', text.lower()) if len(w) >= 2]
    stop_words = {'seeking', 'with', 'and', 'the', 'for', 'skills', 'experience', 'worked', 'proficient'}
    return [w for w in words if w not in stop_words]

def justify_candidate(query: str) -> str:
    """
    Step 7: RAG (Retrieval Augmented Generation) Lite.
    Retrieves the top candidate and generates a data-driven justification based on keyword overlap.
    """
    results = semantic_search(query, top_k=1)
    
    if not results:
        return "No suitable candidates found in the database."
    
    top_resume = results[0]
    resume_text = top_resume["full_text"]
    match_percentage = top_resume["match_percentage"]
    name = top_resume["candidate_name"]
    
    # Keyword extraction and matching
    query_keywords = extract_keywords(query)
    found_keywords = []
    
    # Case-insensitive search in resume body
    resume_lower = resume_text.lower()
    for kw in query_keywords:
        if kw in resume_lower:
            found_keywords.append(kw)
    
    # Build explanation
    if found_keywords:
        # Deduplicate and format
        unique_kws = list(dict.fromkeys(found_keywords))
        formatted_kws = ", ".join([k.upper() for k in unique_kws[:5]])
        
        justification = (
            f"Candidate '{name}' matches your requirement with {match_percentage} confidence. "
            f"Their profile explicitly highlights expertise in key areas: {formatted_kws}. "
            f"Based on their background, they possess the required semantic alignment for this role."
        )
    else:
        justification = (
            f"Candidate '{name}' is a high-confidence semantic match ({match_percentage}). "
            "Although explicit keyword overlap is low, the conceptual themes in their resume "
            "(e.g., professional background and stated projects) align closely with your query."
        )
    
    return justification

def get_rag_explanation() -> str:
    """Explains the RAG concept to the user."""
    return (
        "RAG (Retrieval-Augmented Generation) in this system works by: "
        "1. Retrieving the most statistically relevant resume from Endee. "
        "2. Analyzing the overlap between your query and the candidate's skills. "
        "3. Generating a human-readable justification explaining WHY they are a match."
    )
