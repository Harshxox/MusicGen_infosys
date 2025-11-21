# server.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from musicgen_model import generate_music

app = FastAPI()

# allow frontend JS to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Prompt(BaseModel):
    prompt: str
    duration: float = 15.0

@app.post("/generate")
async def generate_endpoint(data: Prompt):
    try:
        audio_buffer = generate_music(data.prompt, data.duration)
        headers = {"Content-Disposition": 'inline; filename="generated.wav"'}
        return StreamingResponse(audio_buffer, media_type="audio/wav",headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
