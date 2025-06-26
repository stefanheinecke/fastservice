# main.py
import subprocess
import threading
import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

app = FastAPI()

@app.on_event("startup")
def launch_streamlit():
    def run_streamlit():
        subprocess.Popen([
            "streamlit", "run", "app.py",
            "--server.port=8501",
            "--server.baseUrlPath=/streamlit",
            "--server.headless=true",
            "--server.address=0.0.0.0"
        ])
    threading.Thread(target=run_streamlit).start()

@app.get("/api/status")
def read_root():
    return {"message": "Success"}

@app.get("/")
def root_redirect():
    return RedirectResponse("/api/status")
