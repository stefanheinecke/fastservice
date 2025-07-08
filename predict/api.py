from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

app = FastAPI()
@app.get("/sum")
def sum_numbers(a: float = Query(...), b: float = Query(...)):
    result = a + b
    return JSONResponse(content={"sum": result})
