"""Travel Planner — a multi-agent trip planner.

Built on the patterns from the lca-lc-foundations course:

  - 2.4_wedding_planners : coordinator agent + specialist agents wrapped as tools,
                           shared TripState, state written via Command(update=...)
  - 2.1 MCP / 2.1_travel_agent : real flight data via the Kiwi.com MCP server
  - 2.2_state / 2.2_runtime_context : custom state vs per-run context
  - 3.3 / 3.5 : HumanInTheLoopMiddleware gating the booking (side-effect) tools

Models use Gemini (GOOGLE_API_KEY). Hotel/activity search uses Tavily
(TAVILY_API_KEY) and degrades gracefully when that key is not configured yet.
Bookings are SIMULATED: nothing is actually purchased.

Run me:          uv run python smoke_test.py
Deploy me:       Phase 3 (langgraph dev / LangGraph Platform + chat UI)
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, List, Optional

from dotenv import load_dotenv

load_dotenv()

# LangSmith: if the user's configured key is rejected (403) or a placeholder,
# tracing attempts just spam errors. Default to OFF in this codebase, with an
# explicit escape hatch (TRAVEL_PLANNER_TRACING=true) to force it back on.
# The .env file itself is never modified.
_force_tracing = os.environ.get("TRAVEL_PLANNER_TRACING", "").strip().lower() == "true"
if not _force_tracing:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_TRACING"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ.pop("LANGSMITH_API_KEY", None)

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.types import Command
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, TextContent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Defaults to the lite Flash model: it has a far higher free-tier quota than
# gemini-3.5-flash (which is capped at ~20 requests/day on the free tier and
# 429s once exhausted). Override via TRAVEL_PLANNER_MODEL.
MODEL = os.environ.get("TRAVEL_PLANNER_MODEL", "gemini-3.5-flash-lite")
KIWI_MCP_URL = os.environ.get("KIWI_MCP_URL", "https://mcp.kiwi.com")


def make_model() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=MODEL)


# ---------------------------------------------------------------------------
# MCP resilience (pattern from 2.4_wedding_planners)
# ---------------------------------------------------------------------------

RETRYABLE_MCP_CODES = {-32603}


class RetryMCPInterceptor:
    """Retry transient MCP failures with exponential backoff; surface all
    errors to the agent instead of crashing the run."""

    def __init__(self, max_retries: int = 3) -> None:
        self.max_retries = max_retries

    async def __call__(self, request, handler):
        last_error: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            try:
                return await handler(request)
            except McpError as exc:
                last_error = exc
                if exc.error.code not in RETRYABLE_MCP_CODES:
                    return CallToolResult(
                        content=[
                            TextContent(type="text", text=f"Tool call failed (non-retryable): {exc}")
                        ],
                        isError=False,
                    )
            except Exception as exc:  # network / transport failures
                last_error = exc

            if attempt < self.max_retries - 1:
                await asyncio.sleep(2**attempt)

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"Tool call failed after {self.max_retries} attempts: {last_error}",
                )
            ],
            isError=False,
        )


_flight_client: Optional[MultiServerMCPClient] = None
_flight_tools_cache: Optional[List[Any]] = None


async def get_flight_tools() -> List[Any]:
    """Connect to the Kiwi.com MCP server once, return its tools (cached).

    NOTE: for local runs everything happens inside one asyncio event loop,
    which keeps the MCP session happy. Server-side deployment (Phase 3) needs
    a persistent event loop owner for the client.
    """
    global _flight_client, _flight_tools_cache
    if _flight_tools_cache is None:
        _flight_client = MultiServerMCPClient(
            {
                "travel_server": {
                    "transport": "streamable_http",
                    "url": KIWI_MCP_URL,
                }
            },
            tool_interceptors=[RetryMCPInterceptor()],
        )
        tools = await _flight_client.get_tools()
        # Expose only the search tool to the specialist; keep dev feedback out
        # of the model's reach.
        _flight_tools_cache = [t for t in tools if t.name != "feedback-to-devs"]
    return _flight_tools_cache


# ---------------------------------------------------------------------------
# Weather / general web search tool (Tavily)
# ---------------------------------------------------------------------------


@tool
def web_search(query: str) -> str:
    """Search the web for current, factual information (destination guides,
    hotels, activities, attractions, local tips)."""

    key = os.environ.get("TAVILY_API_KEY", "")
    if not key or key.startswith("your_"):
        return (
            "Web search is unavailable: TAVILY_API_KEY is not configured in the .env file. "
            "Tell the user that hotel/activity search needs the Tavily key, then continue "
            "with the best options you can from your own knowledge."
        )

    try:
        from tavily import TavilyClient

        results = TavilyClient().search(query, max_results=5)
        lines = []
        for r in results.get("results", []):
            lines.append(
                f"- {r.get('title')}: {str(r.get('content'))[:400]} ({r.get('url')})"
            )
        return "\n".join(lines) if lines else "No results found."
    except Exception as e:  # noqa: BLE001 - return the error so the model can adapt
        return f"Web search failed: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class TripState(AgentState):
    """Everything the specialists need, gathered by the coordinator."""

    origin: str = ""
    destination: str = ""
    dates: str = ""
    budget: str = ""
    party_size: str = ""
    interests: str = ""
    booking_log: str = ""


# ---------------------------------------------------------------------------
# Booking tools (gated by HumanInTheLoopMiddleware -> must have an explicit
# user confirmation before running. Bookings are simulated.)
# ---------------------------------------------------------------------------


@tool
def book_flight(
    origin: str,
    destination: str,
    airline: str,
    flight_number: str,
    date: str,
    price: str,
    runtime: ToolRuntime,
) -> Command:
    """Book a flight. ONLY call this after the user explicitly confirms they
    want to book this exact option (human approval is required)."""
    entry = (
        f"Flight booked: {airline} {flight_number} | {origin} -> {destination} "
        f"| {date} | {price}"
    )
    previous = runtime.state.get("booking_log") or ""
    return Command(
        update={
            "booking_log": (previous + "\n" + entry).strip(),
            "messages": [
                ToolMessage(content=entry, tool_call_id=runtime.tool_call_id)
            ],
        }
    )


@tool
def book_hotel(
    destination: str,
    hotel: str,
    check_in: str,
    nights: str,
    price: str,
    runtime: ToolRuntime,
) -> Command:
    """Book a hotel stay. ONLY call this after the user explicitly confirms
    they want to book this exact option (human approval is required)."""
    entry = f"Hotel booked: {hotel} | {destination} | check-in {check_in}, {nights} nights | {price}"
    previous = runtime.state.get("booking_log") or ""
    return Command(
        update={
            "booking_log": (previous + "\n" + entry).strip(),
            "messages": [
                ToolMessage(content=entry, tool_call_id=runtime.tool_call_id)
            ],
        }
    )


# ---------------------------------------------------------------------------
# Specialist agents
# ---------------------------------------------------------------------------

FLIGHTS_PROMPT = """\
You are a flight specialist. You use the search-flight tool to find real flight
options between the origin and destination for the requested dates.

Rules:
- No follow-up questions. Use the information you are given.
- Prefer direct (non-stop) flights; mention when only connections exist.
- Use dd/mm/yyyy date format and a currency that matches the user's budget.
- Find the best options by price (lowest) and duration (shortest).
- If the tool fails or returns unusable output, call it again.
- Return the top 3 options. For each: EXACT airline, EXACT flight number,
  departure/arrival times, duration, price, and a short label (e.g.
  "Option 1") so the coordinator can reference it for booking. Never use
  placeholders like "not specified" - every option must include a concrete
  airline and flight number."""

STAYS_PROMPT = """\
You are an accommodation specialist. Use the web_search tool to find good
hotels in the destination that fit the budget and party size.

Rules:
- No follow-up questions.
- Prefer well-reviewed places; note approximate price per night and location.
- List the top 3 options with a short label (e.g. "Option 1") and a rough
  total for the stay if you can estimate it.
- If web search is unavailable, say so and give your best-known options."""

ACTIVITIES_PROMPT = """\
You are an activities specialist. Use the web_search tool to find things to do
in the destination that match the traveller's stated interests and budget.

Rules:
- No follow-up questions.
- Mix famous sights and local gems; flag free vs paid options with rough prices.
- Return the top 5 suggestions grouped by interest, each with a short label.
- If web search is unavailable, say so and give your best-known options."""


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------


def build_travel_planner(checkpointer: Optional[Any] = None):
    """Assemble the coordinator + specialist agents.

    Specialist agents are exposed to the coordinator as tools, so the
    coordinator decides the order of work and merges the results.

    checkpointer: REQUIRED for local scripted runs (interrupts / HITL resume
    need persistence) - pass e.g. InMemorySaver() in smoke tests. Under
    langgraph dev / the platform the server injects its own persistence, so
    the default (None) is what deployment expects.
    """
    stays_agent = create_agent(
        model=make_model(),
        tools=[web_search],
        system_prompt=STAYS_PROMPT,
    )
    activities_agent = create_agent(
        model=make_model(),
        tools=[web_search],
        system_prompt=ACTIVITIES_PROMPT,
    )

    @tool
    async def search_flights(runtime: ToolRuntime) -> str:
        """Delegate flight search to the flight specialist (real Kiwi.com data)."""
        origin = runtime.state.get("origin") or "unknown"
        destination = runtime.state.get("destination") or "unknown"
        dates = runtime.state.get("dates") or "unknown"
        try:
            flight_tools = await get_flight_tools()
            flights_agent = create_agent(
                model=make_model(),
                tools=flight_tools,
                system_prompt=FLIGHTS_PROMPT,
            )
            query = (
                f"Find the best flights from {origin} to {destination} for {dates}. "
                f"Return the top options with airline, flight number, times, "
                f"duration and price."
            )
            resp = await flights_agent.ainvoke(
                {"messages": [HumanMessage(content=query)]}
            )
            return resp["messages"][-1].content
        except Exception as e:  # noqa: BLE001 - graceful degradation
            return (
                f"Flight search failed ({type(e).__name__}: {e}). "
                f"Try again once, then summarise the best you can."
            )

    @tool
    def search_stays(runtime: ToolRuntime) -> str:
        """Delegate hotel search to the accommodation specialist."""
        destination = runtime.state.get("destination") or "unknown"
        budget = runtime.state.get("budget") or "unknown budget"
        party = runtime.state.get("party_size") or "unknown party size"
        resp = stays_agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            f"Find hotels in {destination} for {party}, "
                            f"budget around {budget}."
                        )
                    )
                ]
            }
        )
        return resp["messages"][-1].content

    @tool
    def search_activities(runtime: ToolRuntime) -> str:
        """Delegate activity search to the activities specialist."""
        destination = runtime.state.get("destination") or "unknown"
        interests = runtime.state.get("interests") or "general sightseeing"
        budget = runtime.state.get("budget") or "unknown budget"
        resp = activities_agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            f"Suggest things to do in {destination} matching "
                            f"these interests: {interests}. Budget: {budget}."
                        )
                    )
                ]
            }
        )
        return resp["messages"][-1].content

    @tool
    def update_trip(
        origin: str,
        destination: str,
        dates: str,
        budget: str,
        party_size: str,
        interests: str,
        runtime: ToolRuntime,
    ) -> Command:
        """Save the completed trip brief into shared state. This tool must be
        called ALONE (without other tool calls) so the values become available
        to the other tools before they run."""
        return Command(
            update={
                "origin": origin,
                "destination": destination,
                "dates": dates,
                "budget": budget,
                "party_size": party_size,
                "interests": interests,
                "messages": [
                    ToolMessage(
                        content="Trip brief saved.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    COORDINATOR_PROMPT = """\
You are a travel coordinator. You plan complete trips and ONLY make bookings
after the user explicitly confirms a specific option.

Workflow:
1) Read the user's request and extract: origin, destination, dates, budget,
   party size and interests. When you know ALL of them, call update_trip alone
   (no other tool calls in the same step).
2) Then delegate to your specialists, in any order: search_flights,
   search_stays, search_activities.
3) Merge their findings into a clear day-by-day itinerary with estimated
   costs per part (flights, hotel, activities) and a rough total.
4) Present the itinerary and ask which specific options (if any) the user
   wants you to book.

You must NOT call book_flight or book_hotel unless the user has explicitly
asked to book a specific option. When you do book, pass the EXACT airline,
flight number and date from the search results - never placeholders such as
"not specified". If a specialist reports a failure, note it and continue."""

    return create_agent(
        model=make_model(),
        tools=[
            update_trip,
            search_flights,
            search_stays,
            search_activities,
            book_flight,
            book_hotel,
        ],
        state_schema=TripState,
        checkpointer=checkpointer,  # None = platform-managed persistence (deployment)
        system_prompt=COORDINATOR_PROMPT,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "update_trip": False,
                    "search_flights": False,
                    "search_stays": False,
                    "search_activities": False,
                    "book_flight": True,
                    "book_hotel": True,
                }
            )
        ],
    )


# Exposed for langgraph.json / deployment tooling (Phase 3). Building is
# safe here: no network I/O happens until search_flights is first called.
graph = build_travel_planner()


if __name__ == "__main__":
    print("Travel Planner graph ready:", type(graph).__name__)