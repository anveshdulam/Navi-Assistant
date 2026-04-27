# NavAssist

NavAssist is an AI-powered navigation assistant for visually impaired users. It runs offline with a live camera feed, local object detection, and an optional local SmolVLM2 server for higher accuracy, plus voice alerts, voice commands, and haptic feedback.

## Features
- Live camera feed with auto-scan and manual scan
- Local detection via TensorFlow.js COCO-SSD
- Optional local SmolVLM2 vision (offline)
- Voice alerts and voice commands
- Haptic feedback (mobile)
- Scan history and theme toggle

## Tech stack
- Next.js 16, React 19, Tailwind CSS
- TensorFlow.js + COCO-SSD for local detection
- SmolVLM2 for local vision (offline-first, with optional cloud providers)
- Prisma + SQLite for preferences and scan history

## Quick start
1. Install dependencies:
   - npm install
2. Copy environment file:
   - cp .env.example .env
3. Set required environment variables (see below).
4. Download the local COCO-SSD model for offline use:
   - npm run download:model
5. Generate Prisma client:
   - npm run db:generate
6. (Optional) Create the local database:
   - npm run db:push
7. Start the dev server:
   - npm run dev

Open:http://navi-assistant.vercel.app/

## Environment variables
Offline mode (default):
- OFFLINE_ONLY=true
- NEXT_PUBLIC_OFFLINE_ONLY=true
- NEXT_PUBLIC_COCO_SSD_MODEL_URL=/models/coco-ssd/model.json

Local vision (recommended for SmolVLM2 offline accuracy):
- SMOLVLM2_URL (example: http://127.0.0.1:8081/vision)
- SMOLVLM2_MODEL_ID (optional, used by the local service)

Cloud vision (optional, requires OFFLINE_ONLY=false):
- GEMINI_API_KEY

Optional cloud settings:
- GEMINI_MODEL (default: gemini-2.0-flash)
- ZAI_API_KEY
- ZAI_BASE_URL
- ZAI_VISION_MODEL

Database:
- DATABASE_URL (example: file:./dev.db)

## Local SmolVLM2 service
1. Go to mini-services/smolvlm2_server
2. Create a Python environment and install dependencies:
   - pip install -r requirements.txt
3. Start the service:
   - python app.py
4. Set SMOLVLM2_URL in .env to http://127.0.0.1:8081/vision

Model link:
- https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct

## Voice commands
Examples:
- "start camera"
- "scan now"
- "stop camera"
- "pause scanning"
- "resume scanning"
- "health check"
- "test voice"

## Safety notice
This app is assistive technology and is not a replacement for mobility aids.

## Scripts
- npm run dev
- npm run build
- npm run start (uses Bun; or run: node .next/standalone/server.js)
- npm run lint
- npm run db:generate
- npm run db:push
- npm run db:migrate

## Security
Do not commit .env files or API keys. Use environment variables on your hosting platform.
