# SmolVLM2 Local Vision Service

This service runs SmolVLM2 locally and exposes a simple HTTP endpoint for the app.

## Setup
1. Create a Python environment.
2. Install dependencies:
   - pip install -r requirements.txt
3. Start the server:
   - python app.py

By default, it runs on http://127.0.0.1:8081/vision.

## Environment variables
- SMOLVLM2_MODEL_ID (default: HuggingFaceTB/SmolVLM2-2.2B-Instruct)
- SMOLVLM2_HOST (default: 127.0.0.1)
- SMOLVLM2_PORT (default: 8081)

## API
POST /vision

Body:
{
  "image": "data:image/jpeg;base64,...",
  "prompt": "optional prompt override"
}

Response:
Returns JSON in the app schema:
{
  "scene_summary": "...",
  "objects": [],
  "hazards": [],
  "action_advice": "..."
}
