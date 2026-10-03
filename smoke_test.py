"""End-to-end smoke test for the Travel Planner.

Three scenarios:
  1. Plan a trip (no bookings requested)          -> itinerary, NO interrupt
  2. Ask to book the cheapest flight              -> HITL interrupt fires
  3. Approve the booking via Command(resume=...)  -> booking_log updated

Run from the travel-planner folder:
    uv run python smoke_test.py
"""

import asyncio

# Import travel_planner FIRST: its module-level guard strips the placeholder
# LangSmith key from the process env BEFORE any langchain import can initialize
# the langsmith client (otherwise we get harmless 403 trace spam).
import travel_planner  # noqa: F401

from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from travel_planner import build_travel_planner

TRAVELER = (
    "I'm in London and want a 5-day trip to Paris in mid-June. "
    "Budget around 900 EUR. Solo traveller. "
    "I'm interested in food and museums."
)

# Keep console output ASCII-safe on Windows (cp1252) terminals.
RULE = "-" * 70


async def main() -> None:
    # Local scripted runs need their own checkpointer (the API server injects
    # its own, so the graph default is checkpointer-free).
    agent = build_travel_planner(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "demo-1"}}

    # --- Scenario 1: plan only -------------------------------------------------
    print("=" * 70)
    print("SCENARIO 1 - plan a trip (no bookings requested)")
    print("=" * 70)
    r1 = await agent.ainvoke(
        {"messages": [HumanMessage(content=TRAVELER)]},
        config=config,
    )
    print("Interrupt fired?", bool(r1.get("__interrupt__")), "(expect False)")
    print("-" * 70)
    print(r1["messages"][-1].content[:2000])

    # --- Scenario 2: request a booking -> HITL interrupt ------------------------
    print()
    print("=" * 70)
    print("SCENARIO 2 - ask to book the cheapest flight (HITL should interrupt)")
    print("=" * 70)
    r2 = await agent.ainvoke(
        {"messages": [HumanMessage(content="Please book the cheapest flight option you found.")]},
        config=config,
    )
    interrupt = r2.get("__interrupt__")
    print("Interrupt fired?", bool(interrupt), "(expect True)")
    if interrupt:
        print("-" * 70)
        print("Paused tool call:")
        print(str(interrupt[0].value)[:500])

    # --- Scenario 3: approve the booking ----------------------------------------
    print()
    print("=" * 70)
    print("SCENARIO 3 - human approves the booking (resume)")
    print("=" * 70)
    r3 = await agent.ainvoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config,
    )
    print("Interrupt still pending?", bool(r3.get("__interrupt__")), "(expect False)")
    print("-" * 70)
    print(r3["messages"][-1].content[:800])
    print("-" * 70)
    print("booking_log:")
    print(r3.get("booking_log") or "(empty)")


if __name__ == "__main__":
    asyncio.run(main())