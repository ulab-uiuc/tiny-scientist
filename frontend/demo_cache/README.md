# demo_cache

This folder contains a small mock backend that serves cached `/api/*` responses so you can record a frontend demo without running the real Flask backend.

## Quick start (use the included placeholder cache)

From `frontend/`:

```bash
npm run demo
```

Then open `http://localhost:3030`.

## Regenerate cache from the real backend

1) Start the real backend (port 5000) and make sure it can run end-to-end.

2) Generate cache (still from `frontend/`):

```bash
DEMO_CACHE_API_KEY=... DEMO_CACHE_MODEL=gpt-5-mini \
  python demo_cache/generate_demo_cache.py \
  --backend-base http://localhost:5000 \
  --intent "Self Improving Agents"
```

This overwrites `frontend/demo_cache/cache.json` and stores downloaded artifacts under `frontend/demo_cache/files/`.

## Notes

- The mock backend runs on port `3030` and serves both `/api/*` and the React build from `frontend/build/`.
- `frontend/demo_cache/cache.json` included in git is a placeholder so the demo works out-of-the-box. For a fully faithful demo, regenerate it from the real backend.
