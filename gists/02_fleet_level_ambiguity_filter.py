"""bigquery_agent_analytics SDK — fleet-level ambiguity filter.

Companion Gist for Medium post #1 ("Your BigQuery Agent Analytics
table is a graph. Here's how to see it.") — section 5.

Pulls every trace for a given agent from the last 24 hours and
filters to the ones where a tool call returned multiple matching
contacts — i.e. sessions where the agent had to make a
disambiguation decision.

Install:
    pip install bigquery-agent-analytics

Run:
    python 02_fleet_level_ambiguity_filter.py
"""

from bigquery_agent_analytics import Client, TraceFilter

client = Client(
    project_id="your-project",
    dataset_id="your_dataset",
    table_id="agent_events",
    location="US",
)

traces = client.list_traces(
    filter_criteria=TraceFilter.from_cli_args(
        last="24h",
        agent_id="calendar_assistant",
    )
)

ambiguity_traces = [
    t for t in traces
    if any(
        tc["tool_name"] == "search_contacts"
        and isinstance(tc.get("result"), dict)
        and tc["result"].get("match_count", 0) > 1
        for tc in t.tool_calls
    )
]

print(
    f"{len(ambiguity_traces)} / {len(traces)} traces hit multi-match contact ambiguity"
)
for t in ambiguity_traces:
    print(f"  {t.session_id[:8]} -> {(t.final_response or '')[:80]!r}")
