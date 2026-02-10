# Building Observable AI Agents: Real-Time Analytics for LangGraph with BigQuery

This repository contains the companion code and blog post for building production-grade observability into LangGraph agents using the `BigQueryCallbackHandler` from `langchain-google-community`.

## Contents

| File | Description |
|------|-------------|
| `blog_bigquery_langgraph.md` | Full Medium blog post: step-by-step guide to instrumenting LangGraph agents with BigQuery |
| `demo_travel_agent.py` | Working travel assistant agent with 3 tools that logs all events to BigQuery |
| `blog-post.md` | Blog post: advocating for BigQuery Agent Analytics in Agent Starter Pack |

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

## Agent Starter Pack + BigQuery Agent Analytics

The `blog-post.md` file covers enabling BigQuery Agent Analytics in [Agent Starter Pack](https://github.com/GoogleCloudPlatform/agent-starter-pack) with the `--bq-analytics` CLI flag. It walks through:

- What the ADK plugin captures (event types, schema, token usage)
- Why production agents need structured observability
- Step-by-step setup: project generation, local testing, Cloud Run deployment
- SQL query examples for cost tracking, debugging, and behavior analysis
- Advanced configuration (event filtering, multimodal content, batching)
- Dashboard setup with Looker Studio

**Target audience:** Agent Starter Pack users who want to add production-grade observability to their agents.

## Related resources

- [Agent Starter Pack](https://github.com/GoogleCloudPlatform/agent-starter-pack)
- [ADK Documentation](https://google.github.io/adk-docs/)
- [BigQuery Agent Analytics plugin docs](https://cloud.google.com/bigquery/docs/agent-analytics)
- [BigQuery Agent Analytics Codelab](https://codelabs.developers.google.com/adk-bigquery-agent-analytics-plugin)
- [ADK + BigQuery Codelab](https://codelabs.developers.google.com/codelabs/adk-bigquery-agent-analytics)
- [BigQuery Storage Write API](https://cloud.google.com/bigquery/docs/write-api)
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/)
- [Introducing BigQuery Agent Analytics — Google Cloud Blog](https://cloud.google.com/blog/products/data-analytics/introducing-bigquery-agent-analytics)
