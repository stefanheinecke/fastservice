from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, RedirectResponse
import subprocess
import threading
import httpx

app = FastAPI()

@app.on_event("startup")
def launch_streamlit():
    def run_streamlit():
        subprocess.Popen([
            "streamlit", "run", "app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--server.baseUrlPath=/streamlit",
            "--server.headless=true"
        ])
    threading.Thread(target=run_streamlit, daemon=True).start()

@app.get("/api/status")
def status():
    return {"message": "Success"}

@app.api_route("/streamlit/{path:path}", methods=["GET", "POST"])
async def proxy_streamlit(request: Request, path: str):
    client = httpx.AsyncClient()
    url = f"http://localhost:8501/streamlit/{path}"
    headers = dict(request.headers)

    response = await client.request(
        request.method, url,
        headers=headers,
        content=await request.body()
    )
    return StreamingResponse(
        response.aiter_raw(),
        status_code=response.status_code,
        headers=dict(response.headers)
    )

@app.get("/")
def redirect_root():
    return RedirectResponse("/streamlit")
