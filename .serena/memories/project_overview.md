# Project overview

This Python 3.12 uv workspace builds three independently deployable, single-worker AI voice-assistant services:
- `assistant/`: FastAPI/Uvicorn orchestration of streaming STT, a separately deployed llama.cpp LLM server, tools/MCP, and streaming TTS.
- `stt/`: NeMo 2.4 FastConformer speech-to-text with HTTP and realtime WebSocket APIs.
- `tts/`: Pocket TTS speech synthesis with HTTP streaming, persistent uploaded voice states, and reusable in-memory voice state.
- `tests/`: unittest coverage across all services.
- Shared helpers: `runtime_config.py` and `service_logging.py`.
- `docs/overview.md` is the authoritative architecture/operations overview and must be read before coding.

The assistant admits one utterance at a time. It uses llama.cpp's native `/completion` API, explicit slot IDs, `cache_prompt=true`, stable-prefix prompt reuse, incremental STT-driven cache warming, SSE token streaming, and llama.cpp timing/token metrics. It streams completed sentences to TTS while LLM decoding continues.

All service images listen on port 8080 by default, provide health/config/metrics endpoints, log structured JSON, and reject concurrent inference rather than queueing.