# Building Observable AI Agents: Real-Time Analytics for LangGraph with BigQuery

*How to instrument your LangGraph agents with production-grade observability using the BigQuery Callback Handler*

---

When you deploy an AI agent into production, the first question isn't *"does it work?"* — it's *"how do I know when it stops working?"*

LLM-powered agents are probabilistic systems. They make decisions, call tools, and chain together operations in ways that are difficult to predict. Without observability, debugging a multi-step agent failure is like reading a novel with half the pages torn out. You can't improve what you can't measure.

In this post, I'll walk you through building a LangGraph agent from scratch with full observability powered by Google BigQuery. By the end, you'll have:

- A working multi-tool LangGraph agent powered by Gemini
- Every LLM call, tool execution, and graph transition logged to BigQuery in real time
- SQL queries to analyze latency, trace executions, and debug failures
- Production-ready configuration patterns you can drop into your own project

Let's build it.

---

## The observability gap in LangGraph agents

If you've built LangGraph agents before, you've probably experienced this: your agent works perfectly in development, then fails silently in production. A user reports a wrong answer, but you have no way to reconstruct what happened.

Traditional logging captures inputs and outputs. But LangGraph execution is a *graph* — branching decisions, parallel tool calls, conditional edges. You need to capture the full execution trace with timing data, not just the endpoints.

Consider what happens when your agent handles a single user query:

1. The graph starts and routes to the model node
2. The LLM decides which tools to call (and sometimes calls the wrong ones)
3. Multiple tools execute — possibly in parallel
4. Results flow back to the model node for synthesis
5. The LLM generates a final response

Each of these steps can fail, be slow, or produce unexpected results. Without structured observability, you're flying blind.

---

## Why BigQuery for LangGraph agent analytics?

You might wonder: why not use LangSmith, Langfuse, or another observability platform?

Those are great tools. But BigQuery gives LangGraph users something unique:

- **Serverless scale** — no infrastructure to manage, handles millions of events without provisioning
- **SQL analytics** — query your agent's behavior with tools your data team already knows
- **Native JSON support** — store structured event payloads and query them with `JSON_VALUE()` at read time
- **Real-time streaming** — events land via the [Storage Write API](https://cloud.google.com/bigquery/docs/write-api) within seconds
- **BigQuery ML integration** — use Gemini directly on your logs with `AI.GENERATE()` for automated root cause analysis
- **Unified data platform** — agent logs live alongside your business data in the same warehouse, enabling cross-domain analytics

If you're already on Google Cloud, having your agent's telemetry in the same platform as your application data is a powerful combination. And the `BigQueryCallbackHandler` uses the same `agent_events_v2` schema as the [BigQuery Agent Analytics plugin](https://google.github.io/adk-docs/integrations/bigquery-agent-analytics/) for Google's Agent Development Kit (ADK), so you get a standardized, well-documented event format from day one.

---

## What you'll build

We'll create a **travel assistant agent** that can:
1. Check the current time
2. Look up weather for a city
3. Convert currencies

Every operation — from the initial LLM call to each tool invocation — gets logged to BigQuery with full trace correlation and latency measurements.

Here's the architecture:

```
User Query
    │
    ▼
┌──────────────────────────────────────────┐
│           LangGraph Agent                │
│                                          │
│  ┌─────────┐   ┌───────┐   ┌─────────┐  │
│  │  Model  │──▶│ Tools │──▶│  Model  │  │
│  │  Node   │   │ Node  │   │  Node   │  │
│  └────┬────┘   └───┬───┘   └────┬────┘  │
│       │            │            │        │
│       ▼            ▼            ▼        │
│  ┌──────────────────────────────────┐    │
│  │   BigQuery Callback Handler      │    │
│  │  batching · tracing · latency    │    │
│  └──────────────┬───────────────────┘    │
└─────────────────┼────────────────────────┘
                  │  Storage Write API
                  ▼
     ┌─────────────────────┐
     │    BigQuery Table    │
     │   agent_events_v2   │
     │                     │
     │  SQL Analytics      │
     │  Looker Studio      │
     │  BigQuery ML        │
     └─────────────────────┘
```

---

## Prerequisites

Before we start, make sure you have:

1. **A Google Cloud project** with the BigQuery API enabled
2. **A BigQuery dataset** — the handler creates the table automatically, but the dataset must exist
3. **Authentication** — run `gcloud auth application-default login` locally, or ensure your service account has:
   - `roles/bigquery.jobUser` (project level)
   - `roles/bigquery.dataEditor` (table level)

Keep IAM scoped tightly. The callback handler only needs write access to the events table — don't grant broader permissions than necessary.

Install the dependencies:

```bash
pip install "langchain-google-community[bigquery]" langchain langchain-google-genai langgraph
```

---

## Step 1: Define the tools

Every good agent needs tools. **Tool design directly impacts agent behavior** — clear docstrings and well-typed parameters help the LLM choose the right tool and pass correct arguments.

Let's create three tools for our travel assistant:

```python
from datetime import datetime

from langchain.tools import tool


@tool
def get_current_time() -> str:
    """Returns the current local time."""
    now = datetime.now()
    return f"Current time: {now.strftime('%I:%M:%S %p')} on {now.strftime('%B %d, %Y')}"


@tool
def get_weather(city: str) -> str:
    """Returns the current weather for a specific city."""
    # In production, call a real weather API
    weather_data = {
        "new york": {"temp": 22, "condition": "Clear"},
        "tokyo": {"temp": 24, "condition": "Sunny"},
        "london": {"temp": 14, "condition": "Overcast"},
    }
    city_lower = city.lower()
    if city_lower in weather_data:
        data = weather_data[city_lower]
        return f"Weather in {city.title()}: {data['temp']}°C, {data['condition']}"
    return f"Weather data for '{city}' not available."


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount between currencies."""
    rates = {"USD": 1.0, "EUR": 1.08, "GBP": 1.27, "JPY": 0.0067}
    from_curr, to_curr = from_currency.upper(), to_currency.upper()
    if from_curr not in rates or to_curr not in rates:
        return "Unknown currency"
    result = amount * rates[from_curr] / rates[to_curr]
    return f"{amount} {from_curr} = {result:.2f} {to_curr}"
```

The `@tool` decorator turns regular functions into LangChain-compatible tools with automatic schema generation from type hints and docstrings. Notice how each docstring is a clear, single-sentence description — this is what the LLM reads when deciding which tool to call.

---

## Step 2: Configure the BigQuery Callback Handler

This is where observability begins. The `BigQueryCallbackHandler` intercepts every event in the LangGraph execution lifecycle and streams it to BigQuery.

```python
from langchain_google_community.callbacks.bigquery_callback import (
    BigQueryCallbackHandler,
    BigQueryLoggerConfig,
)

PROJECT_ID = "your-gcp-project-id"
DATASET_ID = "agent_analytics"

# Configure batching and performance
config = BigQueryLoggerConfig(
    batch_size=1,               # Write events immediately (good for dev)
    batch_flush_interval=0.5,   # Flush partial batches every 0.5s
)

# Initialize the handler
handler = BigQueryCallbackHandler(
    project_id=PROJECT_ID,
    dataset_id=DATASET_ID,
    table_id="agent_events_v2",
    config=config,
    graph_name="travel_assistant",  # Enables LangGraph-specific tracking
)
```

A few things to note:

- **`graph_name`** is the key parameter that activates LangGraph integration. Without it, you'll still get LLM and tool events, but you'll miss `NODE_STARTING`, `NODE_COMPLETED`, and graph-level tracking.
- **`batch_size=1`** writes every event immediately. Great for development, but for production you'll want `batch_size=50` or higher to reduce API calls.
- The handler automatically creates the `agent_events_v2` table if it doesn't exist, partitioned by date and clustered by `event_type`, `agent`, and `user_id`.

---

## Step 3: Create the LangGraph agent

Now let's wire everything together into a LangGraph agent:

```python
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Create the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    project=PROJECT_ID,
)

# Create the agent with our tools
tools = [get_current_time, get_weather, convert_currency]
agent = create_agent(llm, tools)
```

`create_agent` creates a ReAct-style agent that reasons about which tools to call, executes them, and synthesizes the results into a final response.

---

## Step 4: Run with graph context tracking

Here's the critical part — wrapping the agent invocation in a `graph_context` to capture the full execution lifecycle:

```python
# Define metadata once to avoid duplication
run_metadata = {
    "session_id": "session-001",
    "user_id": "user-123",
    "agent": "travel_assistant",
}

# Use graph_context for GRAPH_START/GRAPH_END events with latency
with handler.graph_context("travel_assistant", metadata=run_metadata):
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What time is it? What's the weather in Tokyo? "
                    "How much is 100 USD in EUR?"
                )
            ]
        },
        config={
            "callbacks": [handler],
            "metadata": run_metadata,
        },
    )

print(f"Response: {result['messages'][-1].content}")

# Always shut down to flush remaining events
handler.shutdown()
```

The `graph_context` context manager does three important things:

1. **Emits a `GRAPH_START` event** when entering — marking the beginning of a graph execution
2. **Tracks total latency** — measuring wall-clock time for the entire graph run
3. **Emits `GRAPH_END` or `GRAPH_ERROR`** on exit — capturing success or failure with timing

Without `graph_context`, you'll still get individual node and LLM events, but you won't have the graph-level boundary events that let you measure end-to-end execution time.

---

## Step 5: Put it all together

Here's the complete, runnable script:

```python
import os
from datetime import datetime

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_google_community.callbacks.bigquery_callback import (
    BigQueryCallbackHandler,
    BigQueryLoggerConfig,
)
from langchain_google_genai import ChatGoogleGenerativeAI


# --- Tools ---
@tool
def get_current_time() -> str:
    """Returns the current local time."""
    now = datetime.now()
    return (
        f"Current time: {now.strftime('%I:%M:%S %p')} "
        f"on {now.strftime('%B %d, %Y')}"
    )


@tool
def get_weather(city: str) -> str:
    """Returns the current weather for a specific city."""
    weather_data = {
        "new york": {"temp": 22, "condition": "Clear"},
        "tokyo": {"temp": 24, "condition": "Sunny"},
        "london": {"temp": 14, "condition": "Overcast"},
    }
    city_lower = city.lower()
    if city_lower in weather_data:
        data = weather_data[city_lower]
        return (
            f"Weather in {city.title()}: "
            f"{data['temp']}°C, {data['condition']}"
        )
    return f"Weather data for '{city}' not available."


@tool
def convert_currency(
    amount: float, from_currency: str, to_currency: str
) -> str:
    """Convert an amount between currencies."""
    rates = {"USD": 1.0, "EUR": 1.08, "GBP": 1.27, "JPY": 0.0067}
    from_curr = from_currency.upper()
    to_curr = to_currency.upper()
    if from_curr not in rates or to_curr not in rates:
        return "Unknown currency"
    result = amount * rates[from_curr] / rates[to_curr]
    return f"{amount} {from_curr} = {result:.2f} {to_curr}"


# --- Main ---
def main():
    project_id = os.environ.get("GCP_PROJECT_ID", "your-project-id")

    # 1. Configure the callback handler
    config = BigQueryLoggerConfig(
        batch_size=1,
        batch_flush_interval=0.5,
    )

    handler = BigQueryCallbackHandler(
        project_id=project_id,
        dataset_id="agent_analytics",
        table_id="agent_events_v2",
        config=config,
        graph_name="travel_assistant",
    )

    # 2. Create the LLM and agent
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        project=project_id,
    )
    tools = [get_current_time, get_weather, convert_currency]
    agent = create_agent(llm, tools)

    # 3. Run with graph context
    run_metadata = {
        "session_id": "session-001",
        "user_id": "user-123",
        "agent": "travel_assistant",
    }

    query = (
        "What time is it? What's the weather in Tokyo? "
        "How much is 100 USD in EUR?"
    )

    with handler.graph_context(
        "travel_assistant", metadata=run_metadata
    ):
        result = agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={
                "callbacks": [handler],
                "metadata": run_metadata,
            },
        )

    print(f"Response: {result['messages'][-1].content}")

    # 4. Clean up
    handler.shutdown()


if __name__ == "__main__":
    main()
```

Run it:

```bash
export GCP_PROJECT_ID="your-project-id"
python travel_agent.py
```

---

## What gets logged?

After running the agent, your BigQuery table will contain a full execution trace. Here's what we saw from a real run:

| Timestamp | Event Type | Node | Tool | Latency |
|-----------|-----------|------|------|---------|
| 19:57:23 | `GRAPH_START` | — | — | — |
| 19:57:23 | `NODE_STARTING` | model | — | — |
| 19:57:23 | `LLM_REQUEST` | — | — | — |
| 19:57:28 | `LLM_RESPONSE` | — | — | 4,357ms |
| 19:57:28 | `NODE_COMPLETED` | model | — | 4,366ms |
| 19:57:28 | `TOOL_STARTING` | — | get_weather | — |
| 19:57:28 | `TOOL_COMPLETED` | — | get_weather | <1ms |
| 19:57:28 | `TOOL_STARTING` | — | convert_currency | — |
| 19:57:28 | `TOOL_COMPLETED` | — | convert_currency | <1ms |
| 19:57:28 | `TOOL_STARTING` | — | get_current_time | — |
| 19:57:28 | `TOOL_COMPLETED` | — | get_current_time | <1ms |
| 19:57:28 | `NODE_STARTING` | model | — | — |
| 19:57:28 | `LLM_REQUEST` | — | — | — |
| 19:57:33 | `LLM_RESPONSE` | — | — | 5,084ms |
| 19:57:33 | `NODE_COMPLETED` | model | — | 5,089ms |
| 19:57:33 | `GRAPH_END` | — | — | **9,465ms** |

The data tells a clear story: two LLM calls dominate latency (4.4s + 5.1s), while tool execution is sub-millisecond. If we needed to optimize this agent, the LLM calls are the bottleneck — and we'd know exactly where to focus.

Every event includes:
- **`trace_id`** — correlates all events in a single execution
- **`span_id` / `parent_span_id`** — OpenTelemetry-compatible trace hierarchy
- **`latency_ms`** — total time and component breakdown in JSON
- **`attributes`** — LangGraph metadata (node name, execution order, graph name)

### LangGraph-specific event types

The callback handler automatically detects LangGraph execution and emits these events in addition to standard LangChain events:

| Event Type | When It's Emitted |
|---|---|
| `NODE_STARTING` | A LangGraph node begins execution |
| `NODE_COMPLETED` | A LangGraph node completes successfully |
| `NODE_ERROR` | A LangGraph node fails |
| `GRAPH_START` | Graph execution begins (via `graph_context`) |
| `GRAPH_END` | Graph execution completes |
| `GRAPH_ERROR` | Graph execution fails |

This is what sets the BigQuery Callback Handler apart from generic logging — it understands LangGraph's execution model and captures the graph structure, not just individual operations.

---

## Step 6: Analyze your agent with SQL

Now the fun part. With structured events in BigQuery, you can answer questions about your agent's behavior that would be nearly impossible with traditional logging.

### Reconstruct a full execution trace

```sql
SELECT
  timestamp,
  event_type,
  JSON_VALUE(attributes, '$.langgraph.node_name') AS node,
  JSON_VALUE(attributes, '$.tool_name') AS tool,
  JSON_VALUE(latency_ms, '$.total_ms') AS latency_ms
FROM
  `your-project.agent_analytics.agent_events_v2`
WHERE
  trace_id = 'your-trace-id'
ORDER BY
  timestamp ASC;
```

This gives you the complete step-by-step execution flow — invaluable for debugging why an agent took a wrong turn or called the wrong tool.

### Find slow LLM calls

```sql
SELECT
  JSON_VALUE(attributes, '$.model') AS model,
  COUNT(*) AS total_calls,
  ROUND(AVG(CAST(
    JSON_VALUE(latency_ms, '$.total_ms') AS FLOAT64
  )), 0) AS avg_latency_ms,
  ROUND(APPROX_QUANTILES(CAST(
    JSON_VALUE(latency_ms, '$.total_ms') AS FLOAT64
  ), 100)[OFFSET(95)], 0) AS p95_latency_ms
FROM
  `your-project.agent_analytics.agent_events_v2`
WHERE
  event_type = 'LLM_RESPONSE'
  AND DATE(timestamp) = CURRENT_DATE()
GROUP BY
  model;
```

### Analyze tool usage patterns

Understanding which tools your agent uses most — and which ones fail — is critical for optimization. High failure rates on a specific tool might indicate a tool description problem, a schema mismatch, or an upstream API issue:

```sql
SELECT
  JSON_VALUE(attributes, '$.tool_name') AS tool_name,
  COUNTIF(event_type = 'TOOL_COMPLETED') AS successes,
  COUNTIF(event_type = 'TOOL_ERROR') AS failures,
  ROUND(AVG(
    IF(event_type = 'TOOL_COMPLETED',
       CAST(JSON_VALUE(latency_ms, '$.total_ms') AS FLOAT64),
       NULL)
  ), 0) AS avg_latency_ms
FROM
  `your-project.agent_analytics.agent_events_v2`
WHERE
  event_type IN ('TOOL_COMPLETED', 'TOOL_ERROR')
GROUP BY
  tool_name
ORDER BY
  successes DESC;
```

### Track error rates over time

```sql
SELECT
  DATE(timestamp) AS day,
  COUNTIF(status = 'OK') AS successes,
  COUNTIF(status = 'ERROR') AS errors,
  ROUND(
    COUNTIF(status = 'ERROR') * 100.0 / COUNT(*), 2
  ) AS error_rate_pct
FROM
  `your-project.agent_analytics.agent_events_v2`
WHERE
  event_type IN ('GRAPH_END', 'GRAPH_ERROR')
GROUP BY
  day
ORDER BY
  day DESC
LIMIT 30;
```

### Measure graph-level performance by agent

If you run multiple LangGraph agents in production, compare their performance side by side:

```sql
SELECT
  agent,
  COUNT(*) AS total_runs,
  ROUND(AVG(CAST(
    JSON_VALUE(latency_ms, '$.total_ms') AS FLOAT64
  )), 0) AS avg_graph_latency_ms,
  COUNTIF(status = 'ERROR') AS error_count
FROM
  `your-project.agent_analytics.agent_events_v2`
WHERE
  event_type IN ('GRAPH_END', 'GRAPH_ERROR')
  AND DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
GROUP BY
  agent
ORDER BY
  total_runs DESC;
```

### AI-powered root cause analysis

Here's where BigQuery's ML integration shines. You can use Gemini directly on your agent logs to analyze failures — no separate tool required:

```sql
SELECT
  session_id,
  AI.GENERATE(
    (
      'Analyze this agent execution log and explain the root cause of failure: ',
      full_conversation
    ),
    connection_id => 'your-project.us.bqml_connection',
    endpoint => 'gemini-2.5-flash'
  ).result AS root_cause_explanation
FROM
  `your-project.agent_analytics.agent_sessions`
WHERE
  error_message IS NOT NULL
LIMIT 5;
```

This turns raw logs into actionable insights without leaving BigQuery.

---

## Production configuration patterns

### Development: see everything immediately

```python
config = BigQueryLoggerConfig(
    batch_size=1,
    batch_flush_interval=0.5,
    max_content_length=10000,
)
```

### Production: optimize for throughput and cost

Not every event is worth logging. Be deliberate about what you capture to control costs and reduce noise:

```python
config = BigQueryLoggerConfig(
    batch_size=50,               # Batch 50 events per write
    batch_flush_interval=5.0,    # Flush at least every 5 seconds
    queue_max_size=50000,        # Large buffer for traffic spikes
    shutdown_timeout=30.0,       # Extra time to drain on shutdown
    event_allowlist=[            # Only capture what matters
        "LLM_RESPONSE",
        "LLM_ERROR",
        "TOOL_COMPLETED",
        "TOOL_ERROR",
        "GRAPH_END",
        "GRAPH_ERROR",
    ],
)
```

Or if you want most events but need to exclude the noisy ones:

```python
config = BigQueryLoggerConfig(
    event_denylist=["CHAIN_START", "CHAIN_END"],
)
```

### Multimodal agents: offload to GCS

If your agent processes images, PDFs, or generates long outputs, configure GCS offloading. Large content goes to Google Cloud Storage while structured references stay in BigQuery:

```python
config = BigQueryLoggerConfig(
    gcs_bucket_name="my-agent-logs",
    connection_id="us.my-bq-connection",
    max_content_length=500 * 1024,  # 500 KB inline, larger goes to GCS
    log_multi_modal_content=True,
)
```

You can then query the offloaded content directly from BigQuery:

```sql
SELECT
  timestamp,
  part.mime_type,
  STRING(OBJ.GET_ACCESS_URL(part.object_ref, 'r').access_urls.read_url) AS signed_url
FROM `your-project.agent_analytics.agent_events_v2`,
UNNEST(content_parts) AS part
WHERE part.storage_mode = 'GCS_REFERENCE'
ORDER BY timestamp DESC
LIMIT 10;
```

---

## Production table setup

While the handler auto-creates the table, for production you should create it explicitly with the recommended DDL:

```sql
CREATE TABLE `your-project.agent_analytics.agent_events_v2`
(
  timestamp TIMESTAMP NOT NULL,
  event_type STRING,
  agent STRING,
  session_id STRING,
  invocation_id STRING,
  user_id STRING,
  trace_id STRING,
  span_id STRING,
  parent_span_id STRING,
  content JSON,
  content_parts ARRAY<STRUCT<
    mime_type STRING,
    uri STRING,
    object_ref STRUCT<
      uri STRING, version STRING,
      authorizer STRING, details JSON
    >,
    text STRING,
    part_index INT64,
    part_attributes STRING,
    storage_mode STRING
  >>,
  attributes JSON,
  latency_ms JSON,
  status STRING,
  error_message STRING,
  is_truncated BOOLEAN
)
PARTITION BY DATE(timestamp)
CLUSTER BY event_type, agent, user_id;
```

Partition pruning by date and clustering by event type gives you fast filtered queries — critical when you have millions of events.

---

## Going async

If you're running an async application (FastAPI, for example), use the `AsyncBigQueryCallbackHandler`:

```python
import asyncio

from langchain_google_community.callbacks.bigquery_callback import (
    AsyncBigQueryCallbackHandler,
    BigQueryLoggerConfig,
)

handler = AsyncBigQueryCallbackHandler(
    project_id="your-project",
    dataset_id="agent_analytics",
    config=BigQueryLoggerConfig(batch_size=10),
    graph_name="async_assistant",
)

run_metadata = {
    "session_id": "async-session-001",
    "user_id": "user-456",
    "agent": "async_assistant",
}

async with handler.graph_context("async_assistant", metadata=run_metadata):
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="What's the weather?")]},
        config={"callbacks": [handler], "metadata": run_metadata},
    )

await handler.shutdown()
```

The async handler uses `asyncio`-native batching and is safe for concurrent requests — each invocation gets its own trace ID and execution context via `ContextVar`.

---

## Always call `shutdown()`

The handler uses a background thread (or asyncio task) to batch-write events. If your application exits without calling `handler.shutdown()`, you may lose the final batch. In web applications, hook this into your shutdown lifecycle:

```python
import atexit

handler = BigQueryCallbackHandler(...)
atexit.register(handler.shutdown)
```

---

## The build-evaluate-observe cycle

Observability isn't a one-time setup — it's the foundation of continuous improvement. Here's how it fits into the LangGraph agent development lifecycle:

1. **Build** your agent with tools and graph logic
2. **Observe** every execution in production via BigQuery — trace flows, latency, tool usage, errors
3. **Evaluate** by querying your logs — which tools fail most? Where is latency hiding? Are certain user queries triggering bad paths?
4. **Optimize** based on data — improve tool descriptions, adjust system prompts, tune model parameters, add guardrails
5. **Observe again** — verify your changes actually improved things

The SQL queries you write to analyze production logs are the same queries that feed back into your evaluation pipeline. BigQuery becomes both your monitoring system and your data source for continuous agent improvement.

---

## What's next?

Once you have events flowing to BigQuery, the possibilities expand:

- **Connect Looker Studio** for real-time dashboards — there's a [pre-built template](https://lookerstudio.google.com/c/reporting/f1c5b513-3095-44f8-90a2-54953d41b125/page/8YdhF) you can connect to your table
- **Use BigQuery Conversational Analytics** to ask questions about your logs in natural language — "show me error rates by agent this week"
- **Run the [FastAPI monitoring dashboard](https://github.com/langchain-ai/langchain-google/tree/main/libs/community/examples/bigquery_callback/webapp)** for real-time event streaming with Server-Sent Events
- **Set up alerts** with Cloud Monitoring on error rate spikes
- **Try the [ADK BigQuery Agent Analytics Codelab](https://codelabs.developers.google.com/adk-bigquery-agent-analytics-plugin)** for a hands-on tutorial with a multi-agent retail assistant

The BigQuery Callback Handler transforms your LangGraph agents from black boxes into fully observable systems. Every decision, every tool call, every millisecond of latency — captured, queryable, and ready for analysis.

Your agents are only as reliable as your ability to understand what they're doing. Start logging.

---

*The `BigQueryCallbackHandler` is available in [langchain-google-community](https://github.com/langchain-ai/langchain-google/tree/main/libs/community). For the complete API reference and more examples, see the [official documentation](https://docs.langchain.com/oss/python/integrations/callbacks/google_bigquery).*

### Further reading

- [BigQuery Agent Analytics Plugin](https://google.github.io/adk-docs/integrations/bigquery-agent-analytics/) — Official documentation for the shared `agent_events_v2` schema
- [ADK BigQuery Agent Analytics Codelab](https://codelabs.developers.google.com/adk-bigquery-agent-analytics-plugin) — Hands-on tutorial with a multi-agent retail assistant
- [BigQuery Storage Write API](https://cloud.google.com/bigquery/docs/write-api) — How the handler streams events to BigQuery
- [LangGraph Documentation](https://www.langchain.com/langgraph) — LangGraph framework for building agent workflows
