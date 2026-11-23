import uvicorn
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

if __name__ == "__main__":
    print("=" * 70)
    print("Starting HealthAI Integrated Application...")
    print("Open your browser at: http://localhost:8000")
    print("API Documentation:    http://localhost:8000/docs")
    print("=" * 70)
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
