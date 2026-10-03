"""End-to-end HTTP test against the LIVE LangGraph dev server.

Requires the dev server running on http://127.0.0.1:2024 (start it from the
travel-planner folder with `uv run langgraph dev --port 2024 --no-browser`).

Exercises the same REST API the chat UI uses:
    threads -> runs/stream -> HITL interrupt -> resume approval

Run:  uv run python api_smoke_test.py
"""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from langgraph_sdk import get_client

API_URL = os.environ.get("TRAVEL_PLANNER_API_URL", "http://127.0.0.1:2025")
ASSISTANT = os.environ.get("TRAVEL_PLANNER_ASSISTANT", "travel_planner")

TRAVELER = (
    "I'm in London and want a 5-day trip to Paris in mid-June. "
    "Budget around 900 EUR. Solo traveller. "
    "I'm interested in food and museums."
)


async def last_text(state_values: dict) -> str:
    msgs = state_values.get("messages", [])
    for m in reversed(msgs):
        if m.get("type") == "ai" and m.get("content"):
            content = m["content"]
            if isinstance(content, list):  # Gemini block format
                content = "".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            return str(content)[:600]
    return "(no AI message)"


async def run_and_capture(client, thread_id, **kwargs) -> dict:
    final_vals = None
    async for chunk in client.runs.stream(
        thread_id,
        ASSISTANT,
        stream_mode="values",
        **kwargs,
    ):
        # SDK stream v1 yields StreamPart NamedTuples (event, data, id);
        # v2 yields typed dicts. Accept both.
        if isinstance(chunk, tuple):
            event, data = chunk[0], chunk[1]
        else:
            event, data = chunk.get("event"), chunk.get("data")
        if event == "values":
            final_vals = data
    return final_vals or {}


async def main() -> None:
    client = get_client(url=API_URL, api_key=None)

    thread = await client.threads.create()
    tid = thread["thread_id"]
    print("created thread:", tid)

    # --- Run 1: plan only -------------------------------------------------
    print("\n=== RUN 1: plan a trip (no booking) ===")
    vals = await run_and_capture(
        client,
        tid,
        input={"messages": [{"role": "user", "content": TRAVELER}]},
    )
    print("interrupt present?", "__interrupt__" in vals, "(expect False)")
    print("last message:", (await last_text(vals))[:700])

    # --- Run 2: ask to book -> HITL interrupt -------------------------------
    print("\n=== RUN 2: ask to book cheapest flight (expect interrupt) ===")
    vals = await run_and_capture(
        client,
        tid,
        input={
            "messages": [
                {"role": "user", "content": "Please book the cheapest flight option you found."}
            ]
        },
    )
    interrupt = vals.get("__interrupt__")
    print("interrupt present?", interrupt is not None, "(expect True)")
    if interrupt:
        first = interrupt[0]
        value = first.get("value") if isinstance(first, dict) else getattr(first, "value", None)
        action = (value or {}).get("action_requests", [{}])[0]
        print("pending tool:", action.get("name"))
        print("pending args:", action.get("args"))

    # --- Run 3: human approves -----------------------------------------------
    print("\n=== RUN 3: resume with approval ===")
    vals = await run_and_capture(
        client,
        tid,
        input=None,
        command={"resume": {"decisions": [{"type": "approve"}]}},
    )
    print("interrupt still present?", "__interrupt__" in vals, "(expect False)")
    print("booking_log:", vals.get("booking_log") or "(empty)")
    print("last message:", (await last_text(vals))[:400])


if __name__ == "__main__":
    asyncio.run(main())