## Custom GPT Chatbot

Fully custom GPT-2 clone — no OpenAI, no Anthropic. Your model, your server.

## Structure
```
backend/   — Python FastAPI + custom transformer
frontend/  — Next.js chat UI (deploys to Vercel)
```

## Quick Start

### Backend
```bash
cd backend
pip install -r requirements.txt
python train.py
uvicorn api:app --host 0.0.0.0 --port 4000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## Deploy Frontend to Vercel
1. Go to vercel.com → New Project
2. Import `hamad-alharmi/custom-gpt-chatbot`
3. Set root directory to `frontend`
4. Add env var: `NEXT_PUBLIC_API_URL` = your backend URL
5. Deploy

## iOS (PWA — Free)
1. Open the Vercel URL in Safari on iPhone
2. Tap Share → Add to Home Screen
3. Done — works like a native app, no App Store needed

## Backend Hosting (free options)
- **Railway** — free tier, Dockerfile deploy
- **Render** — free tier, Python native
- **Fly.io** — free tier, solid FastAPI support
