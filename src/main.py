"""
Engineering Companion Service - Main Entry Point

This service provides an AI-powered chatbot for engineering students.

To run the service:
    python src/main.py

Or using uvicorn directly:
    uvicorn src.service:app --reload --host 0.0.0.0 --port 8000

API Documentation available at:
    http://localhost:8000/docs
"""

import uvicorn
from src.service import app

if __name__ == "__main__":
    print("=" * 60)
    print("🤖 Engineering Companion Service")
    print("=" * 60)
    print("Starting server...")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
