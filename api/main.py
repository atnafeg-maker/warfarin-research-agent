from fastapi import FastAPI,HTTPException
from agent.schemas import ResearchInput
from agent.engine import evaluate,WARNING
app=FastAPI(title='Warfarin Research Agent — Synthetic Demonstration',version='1.0.0')
@app.get('/health')
def health():return {'status':'ok','mode':'synthetic_demo','warning':WARNING}
@app.post('/demo/predict')
def predict(inp:ResearchInput):
    try:return evaluate(inp)
    except RuntimeError as ex:raise HTTPException(status_code=503,detail=str(ex))
