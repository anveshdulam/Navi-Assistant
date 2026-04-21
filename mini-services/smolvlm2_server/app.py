import base64
import io
import json
import os
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = os.getenv("SMOLVLM2_MODEL_ID", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

app = FastAPI()
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    trust_remote_code=True,
).to(DEVICE)
model.eval()

DEFAULT_PROMPT = (
    "You are NavAssist Vision powered by SmolVLM2 - an expert AI assistant designed specifically for blind and visually impaired users. "
    "Your ONLY job is to analyze the live camera image and describe the real world in front of the user clearly, safely, and concisely so it can be spoken via TTS. "
    "Strict Rules: Detect all major objects, people, obstacles, doors, furniture, stairs, vehicles, text, and hazards. "
    "Explicitly mention walls, closed doors, and large flat surfaces as obstacles when they are visible. "
    "For EVERY object, ALWAYS estimate approximate distance using visual cues (size, perspective, ground reference). "
    "Use ONLY these distance categories: very close (< 1 meter), close (1-2 meters), near (2-4 meters), far (> 4 meters). "
    "Always mention direction: left / center / right / ahead. Be honest - if you are uncertain about distance or identity, say \"uncertain\" instead of guessing. "
    "If the image appears blocked, too dark, too bright, or out of focus, say \"camera view blocked or unclear\" in hazards and action_advice. Do NOT say the path is clear. "
    "Prioritize safety: Always mention hazards, stairs, moving people, or obstacles first. "
    "Keep every description short and natural for voice output (maximum 12-18 words per object). "
    "Do not add extra comments, apologies, or questions. "
    "Output ONLY valid JSON in this exact format: { \"scene_summary\": \"...\", \"objects\": [ { \"name\": \"...\", \"distance\": \"close\", \"direction\": \"right\", \"confidence\": 0.88 } ], \"hazards\": [\"...\"], \"action_advice\": \"...\" }"
)


class VisionRequest(BaseModel):
    image: str
    prompt: Optional[str] = None


def decode_image(data: str) -> Image.Image:
    if data.startswith("data:"):
        try:
            data = data.split(",", 1)[1]
        except IndexError as exc:
            raise HTTPException(status_code=400, detail="Invalid data URL.") from exc
    try:
        raw = base64.b64decode(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image.") from exc
    return Image.open(io.BytesIO(raw)).convert("RGB")


def extract_json(text: str) -> Optional[str]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


@app.post("/vision")
def vision(req: VisionRequest):
    image = decode_image(req.image)
    prompt = req.prompt or DEFAULT_PROMPT

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    prompt_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(text=prompt_text, images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=500)
    text = processor.decode(output[0], skip_special_tokens=True)

    json_str = extract_json(text)
    if not json_str:
        return {
            "scene_summary": "Unable to parse model response.",
            "objects": [],
            "hazards": [],
            "action_advice": "Scan again.",
        }

    try:
        return json.loads(json_str)
    except Exception:
        return {
            "scene_summary": "Unable to parse model response.",
            "objects": [],
            "hazards": [],
            "action_advice": "Scan again.",
        }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("SMOLVLM2_HOST", "127.0.0.1")
    port = int(os.getenv("SMOLVLM2_PORT", "8081"))
    uvicorn.run(app, host=host, port=port)
