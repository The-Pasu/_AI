# AI-Server

Conversation-based risk decision support system with a deterministic pipeline.

## Run

- Install dependencies: `pip install -r requirements.txt`
- Start API: `uvicorn app.main:app --reload`

## API

- Endpoint: POST /api/analyze
- Base URL: http://ai-server:8000
- Request: JSON
- Response: JSON

## Swagger

- http://localhost:8000/docs

## 환경변수

- `.env.example` + 실제 OpenAI API 키
- `OPENAI_OCR_MODEL` (기본값: `gpt-4o-mini`)
- `OCR_DOWNLOAD_TIMEOUT_SECONDS` (기본값: `10`)
- `OCR_MAX_IMAGE_BYTES` (기본값: `5000000`)
