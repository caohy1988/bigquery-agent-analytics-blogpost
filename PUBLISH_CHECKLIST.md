# Publish Checklist: Medium Blog Post #1

> **Companion to**: `medium-blog-1-agent-events-as-graph.md`
>
> **Target URL**: https://medium.com/me/stories (user's drafts)
>
> **Status**: Draft is publication-ready content-wise. This doc tracks the
> remaining non-content items that need resolving before hitting Publish.

---

## 1. Publication target

**Recommendation: Google Cloud Community** (Medium publication handle: `@google-cloud`).

| Option | Pros | Cons | Reach estimate |
|---|---|---|---|
| **Google Cloud Community** (recommended) | Aligned with the product (BQ + ADK); GCP readers are the target audience; co-promotion by the Google Cloud dev-advocacy handles is natural | Submission + editorial review adds ~3–5 business days | ~50k subs, high GCP relevance |
| *The Generator* | Faster editorial turnaround (~1–2 days); broader AI-agent audience; good for cross-vertical brand building | Less targeted — many readers won't be on GCP, some will install the plugin just to try; Google-Cloud-branded content sometimes gets "vendor post" pushback | ~80k subs, mixed technical depth |
| **Personal Medium + GC co-promotion** | Fastest (no editorial gate); preserves author's own Medium stats | Weakest discovery; needs the Google Cloud handle to amplify via reshare, not guaranteed | Personal following + whatever amplification |

Go with **Google Cloud Community**. The post is narrowly targeted at ADK + BQ users, which is that publication's exact audience. Amplification is built in. The 3–5 day editorial turnaround is fine — the post doesn't have a date peg.

**Submission process** (for reference):

1. Paste the article body (between the two `EDITORIAL NOTES — NOT PUBLISHABLE` banners) into a fresh Medium draft under your own account.
2. Add tags: `bigquery`, `ai-agents`, `google-cloud`, `python`, `observability` (Medium max 5).
3. Under *More options → Submit for review*, pick **Google Cloud Community**.
4. Include a short cover note for the publication editor referencing the post's target audience and promising the screenshots/gists are resolved before publish.

---

## 2. Internal DevRel review

Ship this to DevRel before submitting to the publication. One round of eyes catches anything product-positioning-off.

### Draft outreach message

> **Subject**: Medium post review ask — "Your BigQuery Agent Analytics table is a graph"
>
> Hi [DevRel owner],
>
> Draft of the first post in the BQ Agent Analytics SDK / ADK plugin Medium series is ready for review:
>
> - Draft (internal repo): https://github.com/haiyuan-eng-google/bigquery-agent-analytics-blogpost/pull/15 → `medium-blog-1-agent-events-as-graph.md`
> - Target publication: Google Cloud Community
> - Target audience: ADK agent developers + ML platform engineers who have installed the BQ AA plugin and bounced off `agent_events` complexity
> - Primary CTA: install the plugin (5-min quickstart)
> - ~1,630 words of prose, within Medium's sweet spot
>
> Companion assets:
> - Calendar-Assistant demo agent (real, runs against a sandbox GCP project) at `demo_calendar_assistant.py` in the same repo
> - All trace IDs, latencies, and cost numbers in the post are from live runs on Gemini 3 Flash Preview via Vertex AI Express Mode
> - Three Gist-ready code blocks in `gists/` — will be pushed to public Gists before publication
>
> Looking for:
> 1. Product-positioning check (anything I'm accidentally overclaiming or underclaiming about the SDK / plugin?)
> 2. Any red flags on naming the plugin quickstart page as the primary CTA (see item 6 in the draft's open-items list)
> 3. Sign-off on using the sandbox project's real session IDs in the post (they're `test-project-0728-467323` fleet runs, no production data, but wanted to name it)
>
> Happy to walk through the narrative live or iterate async. Tentative submission date: [fill in based on review SLA].
>
> Thanks!
> [Your name]

---

## 3. Primary CTA URL

Section 7's first bullet currently has a `TBD:` marker for the plugin quickstart link. The top-level `adk-python` repo root is explicitly not acceptable (issue #53 framing).

### Candidate URLs

The right URL depends on where the plugin's quickstart lives in the current ADK docs. Two likely candidates:

1. **ADK plugin quickstart page** (preferred if it exists):
   `https://google.github.io/adk-docs/agents/plugins/bigquery-agent-analytics/` *(guess — verify against current adk-docs site structure)*

2. **BQ AA SDK README quickstart section**:
   `https://github.com/GoogleCloudPlatform/BigQuery-Agent-Analytics-SDK#quickstart`
   Fallback if the ADK docs site doesn't yet have a plugin-specific landing page. Anchors directly into the SDK README's quickstart block.

3. **Google Cloud product page** (longest route, worst conversion but highest branding):
   `https://cloud.google.com/bigquery/docs/agent-analytics` *(guess — verify)*

### How to pick

Open each candidate in a browser. Pick the one that:

1. Loads in < 2 seconds (no 404, no full monorepo README to skim).
2. Has a visible "pip install" or equivalent within the first screen.
3. Gets a reader from "I clicked the link" to "I have the plugin running" in ≤ 5 minutes.

Replace the `TBD:` marker in section 7 with the winning URL before submission.

---

## 4. Gists for embedded code blocks

Three inline code blocks in the draft are flagged `<!-- Gist embed candidate: ... -->`. On Medium, Gist embeds render with syntax highlighting and an "Open in GitHub" button that doubles as an SDK backlink.

The content is ready in the `gists/` directory of this repo:

- `gists/01_client_setup_and_render.py` → section 4 featured block
- `gists/02_fleet_level_ambiguity_filter.py` → section 5 fleet-filter block
- `gists/03_information_schema_cost_by_feature.sql` → section 6 cost query

### Create the Gists (manual, ~5 minutes)

Create each at https://gist.github.com/ under the **Google Cloud** account (or the SDK maintainer's) so the "Open in GitHub" link points to an authoritative source, not a personal account:

1. Open https://gist.github.com/
2. Set the Gist description to the filename (e.g. `01_client_setup_and_render.py`).
3. Set the Gist filename to match.
4. Paste the file contents.
5. Create as **public** Gist.
6. Copy the Gist URL.

### Embed in Medium

In the Medium editor, paste each Gist URL on its own line, then press Enter — Medium auto-renders the embed widget. Replace the three inline `<!-- Gist embed candidate: ... -->` blocks with the corresponding embed.

---

## 5. Screenshots

The draft flags four `*[SCREENSHOT: ...]*` placeholders. Medium posts convert better with at least a hero image + one inline image per major section.

### Shot list

| # | Location in draft | Content | How to capture |
|---|---|---|---|
| 1 | Section 1 hook | 47-row `agent_events` query result, messy JSON visible | BQ console → run the query below → crop to show ~8 rows with JSON content column expanded |
| 2 | Section 3 setup | `bq-agent-sdk doctor` output with green checkmarks | Terminal → `bq-agent-sdk doctor` against `test-project-0728-467323` / `agent_analytics_demo` → screenshot the output |
| 3 | Section 4 demo | The featured `trace.render()` output (section 4's code block already contains this as text — screenshot only if you want a visual anchor) | Terminal → `python gists/01_client_setup_and_render.py 84ef108d-745c-451a-ae79-d0f97673268d` |
| 4 | Section 6 cost | `INFORMATION_SCHEMA` query result table | BQ console → run `gists/03_information_schema_cost_by_feature.sql` → screenshot the result pane |
| 5 | Closing | Stylized graphic ("rows → tree") | Designer + original artwork, OR a generated image (`docs/assets/rows_to_tree.png` when created) |

### Exact BQ query for shot #1

```sql
-- Copy into the BQ console for the "raw rows" screenshot.
-- Uses the dataset from section 3 of the post.
SELECT
  timestamp,
  event_type,
  agent,
  SUBSTR(CAST(content AS STRING), 1, 300) AS content_preview,
  status,
  error_message
FROM `test-project-0728-467323.agent_analytics_demo.agent_events`
WHERE session_id = '84ef108d-745c-451a-ae79-d0f97673268d'
ORDER BY timestamp
LIMIT 50;
```

### Screenshot framing guidelines

- Browser: light theme, 100% zoom, no extensions visible in the screenshot area.
- Terminal: dark theme (monospace on dark reads best in Medium's layout), 14pt+ font, window wide enough that no line wraps.
- Crop tightly — no unused chrome, no Finder/Dock visible.
- Export as PNG, not JPG. Medium handles PNG + retina well.
- Save locally as `docs/assets/screenshot_N_<slug>.png` for archival; upload via Medium's image widget at each insertion point.

---

## 6. Optional — Span.summary polish before publication

Section 4's featured `trace.render()` output shows `text: 'I found three people named Priya...'` in the final `LLM_RESPONSE` line. The `text: '...'` wrapper is an artifact of how the agent's final response comes back — not ideal aesthetics for a reader new to the SDK.

This is tracked as a separate SDK PR (pending). If it lands before publication:

1. Re-run the featured session: `python gists/01_client_setup_and_render.py 84ef108d-...`
2. Swap the updated render output into section 4 of the post.

If it doesn't land, publish as-is — the artifact is cosmetic and doesn't undermine the narrative.

---

## Pre-submission final checklist

- [ ] Primary CTA URL resolved (replace `TBD:` in section 7).
- [ ] Three Gists created on the Google Cloud / SDK-owner GitHub account.
- [ ] Gist URLs pasted into Medium draft in place of the three inline code blocks.
- [ ] Five screenshots captured per the shot list; uploaded into the Medium draft at their placeholder positions.
- [ ] Tags added: `bigquery`, `ai-agents`, `google-cloud`, `python`, `observability`.
- [ ] Canonical URL set (if co-publishing on Google Cloud dev blog).
- [ ] Submitted for review to Google Cloud Community publication.
- [ ] DevRel sign-off received.
- [ ] Editorial notes banners stripped from pasted content (top + bottom blocks between the `EDITORIAL NOTES — NOT PUBLISHABLE` markers).
- [ ] (Optional) `Span.summary` polish landed and featured trace re-captured.
