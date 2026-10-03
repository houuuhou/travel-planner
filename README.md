# Travel Planner

A multi-agent trip planner with a streaming chat UI, built on LangChain /
LangGraph patterns from the `lca-lc-foundations` course.

A **coordinator agent** gathers a trip brief into shared `TripState`
(origin, destination, dates, budget, party size, interests), then delegates
to **specialist agents** wrapped as tools:

| Specialist | Data source |
|---|---|
| Flights | **Kiwi.com MCP** (live `search-flight`) + retry interceptor |
| Stays (hotels) | Tavily web search |
| Activities | Tavily web search |

Bookings (`book_flight`, `book_hotel`) are **simulated** and gated behind
`HumanInTheLoopMiddleware` — the agent must present options and the human
must explicitly approve before anything is "booked".

## Quickstart (from scratch)

```bash
# 1. Python deps (uses uv — https://docs.astral.sh/uv/)
uv sync

# 2. Keys
cp .env.example .env            # then fill in GOOGLE_API_KEY (and TAVILY_API_KEY)
cp agent-chat-ui/.env.example agent-chat-ui/.env

# 3. LangGraph API server (no Docker needed — local in-memory runtime)
uv run langgraph dev --port 2025 --no-browser
# -> http://127.0.0.1:2025  (graph: travel_planner)

# 4. Chat UI (Next.js; pnpm via corepack)
cd agent-chat-ui
corepack pnpm install
pnpm dev
# -> http://localhost:3000
```

> Any free port works — `2025` is just what these docs use.

## Smoke tests

In-process (no server):

```bash
uv run python smoke_test.py
```

Exercises the same REST API the UI uses (needs the dev server running):

```bash
uv run python api_smoke_test.py
```

Both cover the three key flows:
1. Plan a trip → itinerary, no interrupt
2. Ask to book a flight → HITL interrupt fires
3. Approve via `Command(resume={"decisions": [{"type": "approve"}]})` → booking logged

## Environment (`.env` at the repo root, loaded automatically)

| Var | Needed for | Status |
|---|---|---|
| `GOOGLE_API_KEY` | the Gemini model | required |
| `TAVILY_API_KEY` | hotel/activity web search | recommended — until set, those specialists report search unavailable and fall back to model knowledge |
| `TRAVEL_PLANNER_MODEL` | override the model (default `gemini-3.5-flash-lite`; full `gemini-3.5-flash` is available but free-tier-capped at ~20 req/day and 429s once hit) | optional |
| `TRAVEL_PLANNER_TRACING=true` | force LangSmith tracing on | optional — leave off unless you have a real LangSmith key |

## Architecture notes

- `search_flights` connects to the Kiwi MCP server lazily (cached) on first
  use, inside whatever event loop is running. Fine for local runs, the smoke
  test, and `langgraph dev`; a production deployment needs a persistent
  event-loop owner for the MCP client.
- **Checkpointer**: `build_travel_planner(checkpointer=None)` by default —
  the LangGraph platform injects its own persistence, so deployable graphs
  must not embed an `InMemorySaver`. Local scripted runs (smoke test) pass
  `checkpointer=InMemorySaver()` explicitly, since interrupts require a
  checkpointer.
- `langgraph.json` is staged for deployment (`travel_planner.py:graph`).
- LangSmith tracing is auto-disabled unless `TRAVEL_PLANNER_TRACING=true`.

## About the chat UI

This repo copies `langchain-ai/agent-chat-ui` into `agent-chat-ui/` (MIT
licensed; see `agent-chat-ui/LICENSE`). Two local adjustments:
- The course snapshot shipped without its `src/lib/` directory; the six
  modules have been restored from upstream (`utils.ts`, `app-config.ts`,
  `api-key.tsx`, `multimodal-utils.ts`, `agent-inbox-interrupt.ts`,
  `ensure-tool-responses.ts`).
- The npm registry mirror `.npmrc` was removed before publishing — if
  `registry.npmjs.org` times out on your network, add a project-local
  `.npmrc` pointing at a mirror (e.g. `registry.npmmirror.com`).

## Known limitations / next steps

- Bookings are simulated — real booking requires provider APIs + payment + HITL.
- Default model is `gemini-3.5-flash-lite` — free-tier quota friendly; the
  full `gemini-3.5-flash` 429s once exhausted. Bump via `TRAVEL_PLANNER_MODEL`.
- Tavily searches occasionally time out (60s) on slow networks — the
  interceptor retries, and the model falls back to knowledge if a retry fails.
- The Kiwi MCP client is built for a single event loop; production server
  deployments need a small refactor (persistent client in a long-running loop).