-- bigquery_agent_analytics SDK — cost by SDK feature.
--
-- Companion Gist for Medium post #1 ("Your BigQuery Agent
-- Analytics table is a graph. Here's how to see it.") —
-- section 6.
--
-- Aggregates BQ jobs labeled by the SDK into a per-feature cost
-- summary: how many runs, total bytes processed, and average
-- slot time per sdk_feature (e.g. trace-read, evaluate-
-- categorical, insights). Every query the SDK runs on your
-- behalf carries labels, so this works without any
-- instrumentation on your side.
--
-- INFORMATION_SCHEMA region must match your dataset's location.
-- The blog-post example uses `region-us` (U.S. multi-region).
-- For single-region datasets, swap in `region-us-central1`,
-- `region-europe-west4`, etc.

SELECT
  (SELECT value FROM UNNEST(labels) WHERE key = 'sdk_feature') AS sdk_feature,
  COUNT(*) AS runs,
  ROUND(SUM(total_bytes_processed) / POW(1024, 3), 3) AS gb_processed,
  ROUND(AVG(total_slot_ms), 0) AS avg_slot_ms
FROM `region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
WHERE creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
  AND EXISTS (
    SELECT 1 FROM UNNEST(labels)
    WHERE key = 'sdk' AND value = 'bigquery-agent-analytics'
  )
GROUP BY sdk_feature
ORDER BY runs DESC;
