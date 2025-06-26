from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, RedirectResponse
import subprocess
import threading
import httpx
import socket
import time

app = FastAPI()


def wait_for_streamlit(host="localhost", port=8501, retries=10, delay=1):
    for i in range(retries):
        try:
            with socket.create_connection((host, port), timeout=1):
                print(f"✅ Streamlit is up on port {port}")
                return
        except OSError:
            print(f"⏳ Waiting for Streamlit... ({i + 1}/{retries})")
            time.sleep(delay)
    print("⚠️ Streamlit failed to start in time")


@app.on_event("startup")
def launch_streamlit():
    def run_streamlit():
        process = subprocess.Popen([
            "streamlit", "run", "app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--server.baseUrlPath=/streamlit",
            "--server.headless=true"
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        # Print output to Cloud Run logs
        for line in process.stdout:
            print("[streamlit]", line.strip())

    threading.Thread(target=run_streamlit, daemon=True).start()
    wait_for_streamlit()

@app.get("/")
def root():
    return {"status": "ok"}  # Health check endpoint

@app.get("/api/status")
def status():
    return {"message": "Streamlit + FastAPI running"}

@app.api_route("/streamlit/{path:path}", methods=["GET", "POST"])
async def proxy_streamlit(request: Request, path: str):
    async with httpx.AsyncClient() as client:
        url = f"http://localhost:8501/streamlit/{path}"
        response = await client.request(
            request.method,
            url,
            headers=dict(request.headers),
            content=await request.body()
        )
        return StreamingResponse(
            response.aiter_raw(),
            status_code=response.status_code,
            headers=dict(response.headers)
        )
