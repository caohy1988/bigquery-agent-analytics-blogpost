"""bigquery_agent_analytics SDK — one-line trace render.

Companion Gist for Medium post #1 ("Your BigQuery Agent Analytics
table is a graph. Here's how to see it.") — section 4.

Install:
    pip install bigquery-agent-analytics

Run:
    python 01_client_setup_and_render.py <session_id>

Embeds into the post via Medium Gist embed widget.
"""

import sys

from bigquery_agent_analytics import Client

client = Client(
    project_id="your-project",
    dataset_id="your_dataset",
    table_id="agent_events",
    location="US",
)

session_id = sys.argv[1] if len(sys.argv) > 1 else "your-session-id"
trace = client.get_session_trace(session_id)
trace.render()

# Structured access, no JSON-digging required.
print("\n--- error spans ---")
print([(s.event_type, s.tool_name, s.error_message) for s in trace.error_spans])

print("\n--- tool calls with ambiguity detection ---")
for c in trace.tool_calls:
    match_count = c["result"].get("match_count") if isinstance(c.get("result"), dict) else None
    print({"tool": c["tool_name"], "args": c["args"], "match_count": match_count})
