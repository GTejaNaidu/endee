"""
Flask frontend for the AI Resume Matching System.

Serves the recruiter dashboard at http://127.0.0.1:5000 and communicates
with the FastAPI backend for search, upload, and match operations.

Environment variables:
    BACKEND_URL    URL of the FastAPI backend (default: http://localhost:8000)
    FLASK_DEBUG    Set to 'true' to enable Flask debug mode (default: off)
"""
import os
import io
import requests
import PyPDF2
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Backend API configuration — override via environment variable in production
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def extract_text(file):
    """Extracts text from PDF or TXT files."""
    filename = file.filename.lower()
    if filename.endswith(".txt"):
        return file.read().decode("utf-8")
    elif filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    return None

@app.route("/")
def index():
    """Home page with search input and upload section."""
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    """Perform search and display results."""
    query = request.form.get("query")
    if not query:
        return redirect(url_for("index"))

    try:
        # Call the FastAPI backend /search endpoint
        response = requests.post(f"{BACKEND_URL}/search", json={"query": query})
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            justification = data.get("rag_justification", "")
            return render_template("results.html", query=query, results=results, justification=justification)
        else:
            error_msg = f"Error from backend: {response.text}"
            return render_template("index.html", error=error_msg)
            
    except Exception as e:
        error_msg = f"Failed to connect to backend: {e}"
        return render_template("index.html", error=error_msg)

@app.route("/upload", methods=["POST"])
def upload():
    """Handle resume file upload and indexing."""
    if "resume" not in request.files:
        return redirect(url_for("index"))
    
    file = request.files["resume"]
    if file.filename == "":
        return redirect(url_for("index"))

    try:
        text = extract_text(file)
        if not text:
            return render_template("index.html", error="Unsupported file format. Use PDF or TXT.")

        # Send text to backend for indexing
        payload = {
            "id": file.filename,
            "text": text
        }
        resp = requests.post(f"{BACKEND_URL}/upload-resume", json=payload)
        
        if resp.status_code == 200:
            return render_template("upload_success.html", filename=file.filename)
        else:
            return render_template("index.html", error=f"Backend upload failed: {resp.text}")

    except Exception as e:
        return render_template("index.html", error=f"Upload error: {e}")

if __name__ == "__main__":
    # Use FLASK_DEBUG=true environment variable to enable debug mode.
    # Never hard-code debug=True — it exposes an interactive debugger in production.
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(port=5000, debug=debug_mode)
