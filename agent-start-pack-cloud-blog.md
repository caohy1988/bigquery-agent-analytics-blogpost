# Track every AI agent interaction with one CLI flag

*BigQuery Agent Analytics gives you a full event-level audit trail for your AI agent — queryable with SQL, one flag to enable.*

---

Imagine the following: a critical production issue where your AI agent is giving users wrong answers. Or a time-sensitive feature launch that needs to ship in weeks. Or a sudden drop in business metrics you can't explain. Or the LLM you depend on gets deprecated tomorrow and you need to swap models fast.

In every one of these scenarios, if you can't see exactly how your agent is behaving, you're flying blind. You need deep visibility — not just logs and traces, but structured data you can actually query and analyze at scale.

**BigQuery Agent Analytics**, now built into [Agent Starter Pack](https://github.com/GoogleCloudPlatform/agent-starter-pack), changes that. It streams every agent interaction into BigQuery as structured event data — giving you a full audit trail you can query with SQL, visualize in dashboards, and analyze with AI. Enabling it takes one CLI flag.

---

## What it captures

Every LLM request, tool call, agent response, and error becomes a row in the `agent_events_v2` table:

| Event Type | Description |
| :---- | :---- |
| `INVOCATION_STARTING` | A new user turn begins |
| `AGENT_STARTING` | The agent starts processing |
| `LLM_REQUEST` | Prompt sent to the model |
| `LLM_RESPONSE` | Model response received |
| `TOOL_STARTING` | Agent calls a tool |
| `TOOL_COMPLETED` | Tool returns a result |
| `AGENT_COMPLETED` | Agent finishes processing |
| `INVOCATION_COMPLETED` | The turn is done |

A simple two-turn conversation with tool calls generates \~22 events, each with timestamps, session IDs, trace IDs, token usage, latency, and the full content payload. Think of it as an event-sourced log for your agent, stored in a warehouse built for analytics at any scale.

---

## What you can do with it

### Debug production issues

When a user reports a bad answer, you can trace the exact sequence — what prompt was sent, which tools were called, what they returned, and what the model decided. Filter by `session_id` to reconstruct any conversation end to end.

```sql
SELECT event_type, agent, timestamp, content, status, error_message
FROM `my_project.my_dataset.agent_events_v2`
WHERE session_id = 'the-problematic-session-id'
ORDER BY timestamp;
```

### Understand behavior at scale

Which tools are used most? Which are failing? How does latency change over time? These are aggregate, business-level questions — exactly what BigQuery excels at. Build queries for tool error rates, latency distributions, and usage trends across your entire agent fleet.

### Cost visibility

LLM tokens cost money. With every request logged, you can track token consumption per user, per session, per model — and spot runaway costs before they hit your bill.

### LLM-powered analysis

Because your data lives in BigQuery, you can use [BigQuery AI functions](https://cloud.google.com/bigquery/docs/ai-application-overview) for semantic clustering, quality scoring, and topic classification — all without moving data. You can even use [Conversational Analytics](https://cloud.google.com/blog/products/data-analytics/introducing-conversational-analytics-in-bigquery) to let anyone on your team ask questions about agent behavior in plain language, no SQL required.

---

## Get started

**Prerequisites:** [Agent Starter Pack](https://github.com/GoogleCloudPlatform/agent-starter-pack) installed (`pipx install agent-starter-pack`), a Google Cloud project with BigQuery API enabled, and `gcloud` authenticated.

Add `--bq-analytics` to your create command:

```shell
agent-starter-pack create my-agent \
  -a adk \
  -d cloud_run \
  --bq-analytics
```

That flag does three things automatically:

- Adds the `google-adk[bigquery-analytics]` dependency to your project  
- Injects the plugin initialization code into `app/agent.py`  
- Configures Terraform infrastructure for BigQuery dataset, GCS bucket, and logging

It works with all ADK-based templates: `adk`, `adk_a2a`, `agentic_rag`, and `adk_live`.

To verify data is flowing, send a few queries through the ADK web UI, then check BigQuery:

```shell
bq query --use_legacy_sql=false \
  "SELECT event_type, agent, timestamp, status
   FROM \`your-project-id.my_agent_analytics.agent_events_v2\`
   ORDER BY timestamp DESC LIMIT 20"
```

---

## Beyond the basics

The default configuration logs everything, but you can customize it for production. Filter which event types get logged, offload multimodal content (images, audio, video) to GCS, and adjust batching thresholds — see the [ADK documentation](https://google.github.io/adk-docs/) for details.

Agent Starter Pack also includes a [pre-built Looker Studio template](https://lookerstudio.google.com/c/reporting/f1c5b513-3095-44f8-90a2-54953d41b125/page/8YdhF) that connects directly to your `agent_events_v2` table for team-wide dashboards covering usage trends, error rates, token consumption, and latency distributions.

Already using Cloud Trace? BigQuery Agent Analytics complements it — Trace handles real-time debugging ("why is this request slow?"), while BigQuery handles offline analysis and BI ("how is my agent performing this week?").

---

## Start building

Add `--bq-analytics` to your next Agent Starter Pack project and turn your agent's black box into a queryable data asset.

- [Agent Starter Pack on GitHub](https://github.com/GoogleCloudPlatform/agent-starter-pack)  
- [ADK documentation](https://google.github.io/adk-docs/)  
- [Google Codelab: BigQuery Agent Analytics](https://codelabs.developers.google.com/adk-bigquery-agent-analytics-plugin)
