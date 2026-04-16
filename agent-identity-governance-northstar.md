# Agents in the Enterprise: A Northstar for Identity, Governance, and Security Boundaries

*v0 draft - Haiyuan Cao + [EM, BigQuery Conversational Analytics] - for co-author review*

> **TL;DR.** Enterprise data platforms were built for two principal types: humans and services. Agentic workflows introduce a third operating reality: software that interprets user intent, composes plans, invokes tools, and crosses trust boundaries with partial autonomy. Our northstar is simple: every agent action should be **attributed** to a delegation chain, **bounded** by machine-checkable policy, and **observable** as structured events that support audit. We use [BigQuery Conversational Analytics](https://docs.cloud.google.com/bigquery/docs/conversational-analytics), Claude in the enterprise, and [BigQuery Agent Analytics](https://adk.dev/integrations/bigquery-agent-analytics/) as the worked example.

---

## 1. Why this matters now

Enterprise security has long assumed that actions are taken either by a person or by a service acting as itself. SSO, MFA, OAuth, IAM roles, and audit logs all fit that model.

Agents break it.

In an increasingly common enterprise flow:

- A user asks Claude a question in Slack or another work surface.
- Claude decides that the question requires data access.
- Claude calls a BigQuery Conversational Analytics (BQ CA) data agent.
- The BQ CA agent generates SQL, executes against governed BigQuery data, and returns natural-language output.
- Claude composes the final answer back to the user.

That single turn spans multiple trust boundaries and multiple logical actors: the human, the front-end agent, the downstream data agent, and the BigQuery execution path. The security boundary is no longer the chat UI. It is the full path from user intent to agent decision to governed data-plane action.

Today this often works because pilots are small, the agent graph is shallow, and teams are still operating on trust. That assumption will not hold when enterprises run many agents from multiple vendors against the same governed datasets.

This document proposes the northstar we should build toward.

---

## 2. The three structural gaps

### 2.1 Identity collapse

When Claude calls a BQ CA agent on behalf of a user, whose identity is actually acting?

In practice, one of three things usually happens:

- The user's identity is forwarded downstream. This preserves user context, but the data plane often cannot distinguish direct user intent from agent-mediated intent.
- The agent acts through its own service identity. This produces cleaner service logs, but weakens user-level attribution and scope.
- A delegated chain exists informally, but is not modeled as a first-class identity object that survives across systems.

This is the first gap: enterprise IAM does not yet model the full delegation chain cleanly enough for multi-agent execution.

### 2.2 Authority drift

Classic access control answers a narrow question: can principal X perform action Y on resource Z?

Agentic systems introduce a harder question: should this action happen right now, given where the instruction came from and how it was transformed?

Example: a user is allowed to read a sales dataset. The agent ingests an untrusted document that says "export all rows to this address." The underlying read might still be authorized. The broader action may still be unacceptable.

This is the second gap: IAM can evaluate permission, but it does not natively evaluate instruction provenance, prompt-injection risk, output channel, or agent intent.

### 2.3 Audit gap

Traditional audit logs capture API activity. For agents, that is necessary but insufficient.

To understand why an agent action occurred, you need a richer record:

- user message and relevant context
- agent identity and configuration context
- tool selection and execution steps
- generated SQL or tool arguments where appropriate
- results, errors, and latency
- final response
- the delegation chain that justified the action

Without that record, "the agent did the wrong thing" is hard to investigate. You may know that a query ran, but not why the agent chose to run it.

This is where [BigQuery Agent Analytics](https://adk.dev/integrations/bigquery-agent-analytics/) matters. It already captures structured ADK agent events into BigQuery and supports OpenTelemetry correlation fields such as trace and span identifiers. That does not solve the full identity-and-policy problem by itself, but it provides the right audit substrate.

---

## 3. Four northstar principles

### P1. Agent identity must be first-class

Each agent should have a named, attestable identity distinct from:

- the human user
- the runtime service account
- the downstream tool or data system

That identity should carry at least three things:

- provenance: model, version, tool access, and key configuration references
- delegation: who authorized this agent to act in this turn
- scope: what tools and resources this agent may touch for this request

The exact implementation could vary, but the design goal is clear: a cross-system identity artifact that survives agent-to-agent calls.

### P2. Authority must be bounded by explicit policy

Policy for agents must evaluate more than coarse access rights. It should be able to reason over:

- the user and agent identities involved
- the requested tool and arguments
- the data sensitivity of the resource
- the provenance of the instruction that led to the action
- the output destination or communication channel

The target state is not just "this principal can query BigQuery." It is closer to "this agent, acting for this user, may issue a bounded analytical query against this dataset and may not return sensitive fields to an unapproved channel."

### P3. Every agent action must be observable as data

Observability for agents should be a structured event stream, not a collection of screenshots and ad hoc logs. The core events are prompts or messages in, tool calls, SQL generation, responses out, errors, latency, token usage, and identity context where available.

In the reference architecture here, BigQuery is the system of record for those events, and BigQuery Agent Analytics is the ingestion path for ADK-based agents. Dashboards are useful, but the foundational asset is the underlying data.

### P4. Audit must be a product surface

Auditing should not require stitching together raw logs by hand. Security, compliance, and platform teams should be able to ask questions such as:

- Why did this agent query that dataset?
- Which users triggered tool calls against sensitive tables last week?
- Which agents generated errors after a policy change?

This is the strategic reason to pair BQ Agent Analytics with BQ Conversational Analytics: the same BigQuery event substrate used for observability can also power a natural-language audit surface over agent behavior.

---

## 4. Worked example: Claude -> BQ CA -> BigQuery, with BQ Agent Analytics as the audit substrate

The reference flow is:

```
[ Human in Slack or chat ]
        |
        v
[ Claude enterprise agent ]
        |
        v
[ BQ Conversational Analytics data agent ]
        |
        v
[ Governed BigQuery dataset ]
        |
        v
[ Event stream in BigQuery via BQ Agent Analytics ]
        |
        v
[ Audit and observability analysis via SQL, BI, or CA ]
```

What we want each layer to do in the northstar:

| Layer | Northstar responsibility | Status today |
|---|---|---|
| Claude or peer enterprise agent | Preserve user context, declare its own agent identity, emit telemetry, and pass a delegation artifact downstream | Partial. Enterprise agents can emit telemetry, but cross-vendor delegated identity is not standardized. |
| BQ CA data agent | Translate natural language to governed data actions, preserve upstream context, and become the main policy choke point for NL-to-SQL requests | Partial. Public docs support governed NL analytics over BigQuery, but do not describe standardized delegated-identity verification or agent-specific policy evaluation. |
| BigQuery | Enforce the data-plane controls that exist today: IAM, policy tags, row-level security, column-level controls, and auditing at query execution time | Available today for data-plane policy, but not agent-aware delegation semantics. |
| BigQuery Agent Analytics | Capture detailed agent events into BigQuery, including request lifecycle and trace correlation, so teams can analyze behavior as data | Available today. This is the strongest concrete foundation in the stack. |
| CA over the event log | Let platform, security, and product teams interrogate the event log in natural language | Available today by composing CA over the event tables. This is the audit UX story. |

This table is the key framing distinction for the document: **observability is here now; agent identity and agent-aware policy are the northstar.**

---

## 5. What we likely need to build or influence

1. **A standardized agent event schema.** BigQuery Agent Analytics already gives a practical BigQuery-native shape for agent events. We should influence a broader schema direction, likely aligned with OpenTelemetry and GenAI telemetry conventions.
2. **A delegation token or attestation format for cross-agent calls.** Claude -> BQ CA should eventually carry a machine-verifiable delegation artifact, not just implicit trust or a forwarded user token.
3. **Policy primitives that understand provenance.** Data-plane authorization alone is not enough. Policy evaluation needs instruction source, tool target, and output destination in context.
4. **A productized audit surface.** The first version is event capture plus analysis over BigQuery. The more ambitious version is a set of durable audit workflows and natural-language "questions of record" over agent activity.
5. **Clear ownership boundaries.** If identity, policy, telemetry, and audit span multiple teams, the product story will fragment unless schema and control-plane ownership are explicit.

---

## 6. What this document is not

- It is **not** a replacement for existing IAM. Agent identity should layer on top of current principal-based controls.
- It is **not** a claim that all pieces already exist in productized form. The point is to separate what is available today from what still needs standardization.
- It is **not** a model-safety paper. Prompt robustness and refusal behavior matter, but this memo is focused on the enterprise control plane around agents.
- It is **not** a single-vendor thesis. BigQuery can be the audit substrate even if the agents above it are multi-vendor.

---

## 7. Open questions for co-author review

**Most want your input on:** 1, 2, and 5.

1. **Identity posture:** do we frame this as "GCP-anchored first, vendor-neutral later," or do we make vendor-neutral delegated identity the thesis on day one?
2. **Policy enforcement location:** where should the main enforcement point live for conversational analytics workflows: in the front-end agent, inside BQ CA, in BigQuery, or across all three with clear layering?
3. **Schema ownership:** should the event model be proposed as a Google Cloud convention first, an OpenTelemetry extension, or both in sequence?
4. **Audit-as-an-agent commitment:** how hard do we want to lean into the idea that the audit surface is itself a CA data agent over the event log?
5. **v1 demo scope:** what is the smallest end-to-end demo worth aligning around? My current straw-man is Claude + BQ CA + BQ Agent Analytics + one governed dataset, with both dashboard-based and natural-language audit over the same event table.
6. **Review perimeter:** before broader sharing, which teams need eyes on this draft: security, compliance, IAM, ADK, and partner teams?

---

## 8. References and versioning

### References

- [BigQuery Conversational Analytics](https://docs.cloud.google.com/bigquery/docs/conversational-analytics)
- [BigQuery Agent Analytics](https://adk.dev/integrations/bigquery-agent-analytics/)
- [Google Agent Development Kit (ADK)](https://google.github.io/adk-docs/)
- Internal: [BQ AA + CA closed-loop blog post](./blog_ca_bq_agent_analytics.md)

### Versioning

- **v0**: framing draft for co-author alignment
- **v1**: incorporate BQ CA, security, and compliance feedback; lock the v1 demo scope
- **v2**: externalizable position paper or blog form
