# Bhagavatam AI: web UI

Next.js 16 + React 19 + TypeScript + Tailwind CSS 4 chat interface for the Bhagavatam AI API.

The browser only talks to this server. Requests to `/api/*` are proxied to the FastAPI backend at
`BHAGAVATAM_API_URL` (default `http://127.0.0.1:8000`, see `next.config.ts`), so one port is enough
locally, in GitHub Codespaces and behind a reverse proxy.

```bash
npm ci
npm run dev        # http://localhost:3000 (start the API first: `make api` in the repository root)
npm run lint && npm run typecheck && npm test && npm run build
```

Conversations are stored in the browser's localStorage only. The server does not keep them.
