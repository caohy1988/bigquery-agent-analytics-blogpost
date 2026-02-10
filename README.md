# Building Observable AI Agents: Real-Time Analytics for LangGraph with BigQuery

This repository contains the companion code and blog post for building production-grade observability into LangGraph agents using the `BigQueryCallbackHandler` from `langchain-google-community`.

## Contents

| File | Description |
|------|-------------|
| `blog_bigquery_langgraph.md` | Full Medium blog post: step-by-step guide to instrumenting LangGraph agents with BigQuery |
| `demo_travel_agent.py` | Working travel assistant agent with 3 tools that logs all events to BigQuery |

## Quick start

### Prerequisites

- Python 3.10+
- A Google Cloud project with BigQuery enabled
- Application Default Credentials configured (`gcloud auth application-default login`)

### Install dependencies

```bash
pip install langchain-google-community[bigquery] langchain-google-genai langgraph
```

### Run the demo

```bash
export GCP_PROJECT_ID="your-project-id"
python demo_travel_agent.py
```

The agent will:
1. Answer three questions (current time, Tokyo weather, USD-to-EUR conversion)
2. Log all events (graph, node, LLM, tool) to BigQuery in real time
3. Print a `bq query` command you can run to inspect the logged events

### Query your events

```sql
SELECT
  timestamp,
  event_type,
  JSON_VALUE(latency_ms, '$.total_ms') AS latency_ms
FROM `your-project.agent_analytics.agent_events_v2`
WHERE DATE(timestamp) = CURRENT_DATE()
ORDER BY timestamp
```

## What you'll learn

- How to configure `BigQueryCallbackHandler` with batching, event filtering, and GCS offloading
- How `graph_context()` correlates events across an entire graph execution
- SQL queries for latency analysis, tool usage patterns, error detection, and multi-agent comparison
- Production configuration patterns for high-throughput deployments
- Async support with `AsyncBigQueryCallbackHandler`

## Related resources

- [BigQuery Agent Analytics plugin docs](https://cloud.google.com/bigquery/docs/agent-analytics)
- [ADK + BigQuery Codelab](https://codelabs.developers.google.com/codelabs/adk-bigquery-agent-analytics)
- [BigQuery Storage Write API](https://cloud.google.com/bigquery/docs/write-api)
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/)
