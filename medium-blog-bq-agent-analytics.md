# Your AI Agent Is Live. Do You Know What It's Doing?

*One CLI flag turns your Agent Starter Pack project into a fully observable system — with every interaction queryable in BigQuery.*

---

You shipped your AI agent. Users are talking to it. Stakeholders are excited.

Then someone asks: "Why did the agent tell a customer to contact a department that doesn't exist?" And you realize you have no way to find out.

You can't see what prompt was sent. You can't see which tools the agent called. You can't tell if the model hallucinated or if a tool returned bad data. You're debugging a probabilistic system with `print` statements and hope.

**This is the observability gap** — and it's where most AI agent projects hit a wall after launch.

The [Agent Starter Pack](https://github.com/GoogleCloudPlatform/agent-starter-pack) now ships with a built-in solution: **BigQuery Agent Analytics**. It streams every agent event — every LLM call, every tool execution, every error — into a BigQuery table. Enabling it takes one flag. Querying it takes SQL you already know.

---

## What happens when you flip the switch

Add `--bq-analytics` to your project creation command:

```bash
agent-starter-pack create my-agent \
  -a adk \
  -d cloud_run \
  --bq-analytics
```

That single flag triggers three things automatically:

1. **Adds the dependency** — `google-adk[bigquery-analytics]>=1.21.0` is added to your project
2. **Injects the plugin code** — your `app/agent.py` gets the BigQuery Agent Analytics plugin wired in
3. **Generates infrastructure wiring** — Terraform configs and environment variable mappings for BigQuery dataset, GCS bucket, and connection are included in the generated project (provisioned when you run Terraform or deploy)

No manual setup. No config files to edit. The flag works with three ADK-based templates: **`adk`**, **`adk_a2a`**, and **`agentic_rag`**.

---

## What gets captured

Every step of your agent's execution becomes a row in the `agent_events_v2` table:

| Event Type | What It Means |
|---|---|
| `INVOCATION_STARTING` | A new user turn begins |
| `LLM_REQUEST` | Prompt sent to the model |
| `LLM_RESPONSE` | Model response received |
| `TOOL_STARTING` | Agent calls a tool |
| `TOOL_COMPLETED` | Tool returns a result |
| `AGENT_COMPLETED` | Agent finishes processing |
| `INVOCATION_COMPLETED` | The turn is done |

A simple two-turn conversation with tool calls generates ~22 events. Each carries timestamps, session IDs, trace IDs, token counts, latency measurements, and the full content payload.

Think of it as **event sourcing for your agent** — a complete, replayable record of every decision it made and why.

---

## The code that makes it work

Here's a trimmed excerpt of the generated `app/agent.py` with the plugin enabled (the actual template uses cookiecutter variables for the app name and directory):

```python
from google.adk.plugins.bigquery_agent_analytics_plugin import (
    BigQueryAgentAnalyticsPlugin,
    BigQueryLoggerConfig,
)
from google.cloud import bigquery

_plugins = []
_project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
_dataset_id = os.environ.get("BQ_ANALYTICS_DATASET_ID", "adk_agent_analytics")
_location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

if _project_id:
    try:
        bq_client = bigquery.Client(project=_project_id)
        bq_client.create_dataset(f"{_project_id}.{_dataset_id}", exists_ok=True)

        _plugins.append(
            BigQueryAgentAnalyticsPlugin(
                project_id=_project_id,
                dataset_id=_dataset_id,
                location=_location,
                config=BigQueryLoggerConfig(
                    gcs_bucket_name=os.environ.get("BQ_ANALYTICS_GCS_BUCKET"),
                    connection_id=os.environ.get("BQ_ANALYTICS_CONNECTION_ID"),
                ),
            )
        )
    except Exception as e:
        logging.warning(f"Failed to initialize BigQuery Analytics: {e}")

app = App(
    root_agent=root_agent,
    name="app",
    plugins=_plugins,
)
```

A few things to note:

- **Graceful degradation** — if the plugin fails to initialize (missing permissions, network issues), the agent still runs normally. Analytics just won't be captured.
- **Environment-driven config** — all settings come from environment variables, so you can tune behavior per environment without code changes.
- **Auto-creates the dataset** — no need to pre-provision BigQuery resources for local development.

---

## See it working in 60 seconds

Set your environment variables and start the playground:

```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
export BQ_ANALYTICS_DATASET_ID=my_agent_analytics

make install && make playground
```

Send a few messages through the ADK web UI, then query BigQuery:

```bash
bq query --use_legacy_sql=false \
  "SELECT event_type, agent, timestamp, status
   FROM \`your-project-id.my_agent_analytics.agent_events_v2\`
   ORDER BY timestamp DESC LIMIT 20"
```

You should see a full event trail for your conversations — every LLM call, tool invocation, and completion status, all in structured rows.

---

## Four things you can do on day one

### 1. Debug that bad answer

A user reports the agent gave wrong information. Trace the exact sequence:

```sql
SELECT event_type, agent, timestamp, content, status, error_message
FROM `my_project.my_dataset.agent_events_v2`
WHERE session_id = 'the-problematic-session-id'
ORDER BY timestamp;
```

Here's what a real trace looks like for a single user turn where the agent called a tool:

```
+----------------------+------------+---------------------+-----------+
| event_type           | agent      | timestamp           | status    |
+----------------------+------------+---------------------+-----------+
| INVOCATION_STARTING  | root_agent | 2026-02-10 20:25:27 | OK        |
| LLM_REQUEST          | root_agent | 2026-02-10 20:25:27 | OK        |
| LLM_RESPONSE         | root_agent | 2026-02-10 20:25:29 | OK        |
| TOOL_STARTING        | root_agent | 2026-02-10 20:25:29 | OK        |
| TOOL_COMPLETED       | root_agent | 2026-02-10 20:25:30 | OK        |
| LLM_REQUEST          | root_agent | 2026-02-10 20:25:30 | OK        |
| LLM_RESPONSE         | root_agent | 2026-02-10 20:25:32 | OK        |
| AGENT_COMPLETED      | root_agent | 2026-02-10 20:25:32 | OK        |
| INVOCATION_COMPLETED | root_agent | 2026-02-10 20:25:32 | OK        |
+----------------------+------------+---------------------+-----------+
```

Read this top to bottom and you see the full story: the user's turn starts (`INVOCATION_STARTING`), the agent sends a prompt to the model, gets a response that includes a tool call, executes the tool, sends the tool result back to the model for a second LLM round-trip, and completes. Total wall time: ~5 seconds. If something went wrong — a `TOOL_ERROR` status, an unexpected second tool call, a suspiciously long gap between timestamps — you'd spot it immediately.

The `content` column (omitted above for readability) contains the actual payloads: the full prompt text, the model's response, the tool arguments and return values. That's where you go to answer "what exactly did the model see and decide?"

### 2. Track token costs

LLM tokens cost money. Find out where it's going:

```sql
SELECT
  agent,
  JSON_VALUE(attributes, '$.model') AS model,
  SUM(CAST(JSON_VALUE(attributes, '$.usage_metadata.prompt') AS INT64)) AS prompt_tokens,
  SUM(CAST(JSON_VALUE(attributes, '$.usage_metadata.completion') AS INT64)) AS completion_tokens
FROM `my_project.my_dataset.agent_events_v2`
WHERE event_type = 'LLM_RESPONSE'
GROUP BY agent, model
ORDER BY prompt_tokens DESC;
```

### 3. Monitor tool reliability

Which tools are failing, and how often?

```sql
SELECT
  JSON_VALUE(content, '$.tool') AS tool_name,
  COUNT(*) AS call_count,
  COUNTIF(status = 'ERROR') AS error_count,
  ROUND(COUNTIF(status = 'ERROR') / COUNT(*) * 100, 2) AS error_rate_pct
FROM `my_project.my_dataset.agent_events_v2`
WHERE event_type IN ('TOOL_COMPLETED', 'TOOL_ERROR')
GROUP BY tool_name
ORDER BY call_count DESC;
```

### 4. Analyze conversations with AI

Because your data lives in BigQuery, you can use **Gemini-backed AI functions** like `AI.GENERATE` and `AI.CLASSIFY` to analyze agent behavior directly in SQL — classify user intents, scan for PII in responses, run root cause analysis on failed sessions. For time-series use cases, BigQuery ML's `AI.DETECT_ANOMALIES` can flag unusual latency patterns. All without moving data out of BigQuery.

And with [**Conversational Analytics**](https://cloud.google.com/bigquery/docs/conversational-analytics), anyone on your team can ask questions about agent behavior in plain English — no SQL required.

---

## Going to production

### Deploy to Cloud Run

```bash
gcloud run deploy my-agent \
  --source . \
  --memory 4Gi \
  --region us-central1 \
  --no-allow-unauthenticated \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=your-project-id,BQ_ANALYTICS_DATASET_ID=my_agent_analytics"
```

If you generated with CI/CD (`--cicd-runner google_cloud_build`), just push to your repo. The pipeline deploys automatically and Terraform provisions all the BigQuery infrastructure.

### Tune for production workloads

The default configuration logs everything, which is ideal for development. In production, you'll want to be selective:

```python
config = BigQueryLoggerConfig(
    # Only capture what matters
    event_allowlist=[
        "LLM_RESPONSE", "LLM_ERROR",
        "TOOL_COMPLETED", "TOOL_ERROR",
    ],
    batch_size=50,  # Buffer before writing
)
```

For multimodal agents handling images or audio, offload large content to GCS:

```python
config = BigQueryLoggerConfig(
    gcs_bucket_name="my-agent-logs-bucket",
    connection_id="us-central1.my-bq-connection",
    log_multi_modal_content=True,
    max_content_length=500 * 1024,  # Offload content > 500KB
)
```

### Build dashboards

Your `agent_events_v2` table works well with **Looker Studio** — connect it as a BigQuery data source to build dashboards covering usage trends, tool error rates, token consumption, and latency distributions. You can also use Grafana, Metabase, or any BI tool that connects to BigQuery.

---

## How it fits with what you already have

Agent Starter Pack already gives you Cloud Trace for real-time debugging. BigQuery Agent Analytics is the complementary piece for **offline analysis and business intelligence**:

| | Cloud Trace | BigQuery Agent Analytics |
|---|---|---|
| **Best for** | "Why is this request slow right now?" | "How is my agent performing this week?" |
| **Data format** | Spans and traces | Structured event rows |
| **Query with** | Trace Explorer UI | SQL |
| **Always on?** | Yes | Opt-in (`--bq-analytics`) |

Together, they give you full-stack observability: real-time debugging and long-term analytics in one project.

---

## Get started

1. Install Agent Starter Pack: `pipx install agent-starter-pack`
2. Create a project with analytics: `agent-starter-pack create my-agent -a adk --bq-analytics`
3. Set `GOOGLE_CLOUD_PROJECT` and run `make playground`
4. Send some messages, then query `agent_events_v2` in BigQuery

Your agent shouldn't be a black box. With one flag, it isn't.

---

*Links: [Agent Starter Pack on GitHub](https://github.com/GoogleCloudPlatform/agent-starter-pack) | [ADK documentation](https://google.github.io/adk-docs/) | [BigQuery Agent Analytics Codelab](https://codelabs.developers.google.com/adk-bigquery-agent-analytics-plugin)*
