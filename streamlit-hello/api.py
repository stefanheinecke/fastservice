from fastapi import FastAPI

app = FastAPI()

@app.get("/api/status")
def read_root():
    return {"message": "Success"}