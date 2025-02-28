# File structure:
# - app.py (main application)
# - scraper.py (documentation scraper)
# - processor.py (query processing)
# - templates/index.html (web interface)
# - static/style.css (styling)
# - static/script.js (frontend logic)
# - requirements.txt (dependencies)

# app.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
from processor import QueryProcessor
import os

app = FastAPI(title="CDP Documentation Chatbot")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Initialize the query processor
processor = QueryProcessor()

class Query(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/query")
async def process_query(query: Query):
    try:
        response = processor.process_query(query.question)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/refresh_docs")
async def refresh_docs():
    try:
        processor.refresh_documentation()
        return {"status": "Documentation refreshed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)