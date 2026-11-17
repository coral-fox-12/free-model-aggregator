# Free Model Aggregator

Route any prompt to the best **free** small model available at $0, with
deterministic failover and a public audit trail of what happened.

One server, two client wires. It speaks both **OpenAI Chat Completions**
(`POST /v1/chat/completions`) and **Anthropic Messages**
(`POST /v1/messages`), translating between them, so any existing tool can
point at it without changes. A request names model `"auto"`; the server
replaces it with the best-rated free model that meets the prompt's
requirements (JSON mode, tool use, vision), dials it through the gateway,
and returns the answer in the wire the caller used — streamed or not, with
usage and stop reason intact.

## Why it exists

Frontier AI subscriptions are unaffordable for many of the people who
could use AI most: students, independent developers, teachers, and small
nonprofits in Vietnam and other resource-constrained regions. The free
tiers of small open models already answer ordinary tasks well — drafting,
Q&A, JSON extraction, tool calls — but any single free tier rate-limits,
goes down, or returns an unusable answer. This server makes free models
reliable by routing across all of them and failing over transparently.

## How failover works

Every prompt is served by the best-rated free model that meets its
requirements. When a gateway call is refused (429 / 5xx / network error)
or returns an unusable answer (no token, no text, bad JSON, unreadable
stream), the next-best candidate serves the prompt instead. Every refusal
is reported to the caller on the `x-makefun-upstream-failures` response
header — a JSON array of `{status, code, model}` rows in the order the
calls refused — so reliability is inspectable, not marketed. An answer
only counts when the provider's answer body carries the prompt's token.

## Models

The catalog is the free tier of the gateway (OpenRouter), all priced $0.
The routing set, with the house quality rating that orders it:

| Model | Rating |
|---|---|
| thinkingmachines/inkling-2-small | 1475 |
| nvidia/nemotron-3.5-super-vl | 1303 |
| nvidia/nemotron-3-ultra-550b-a55b | 1285 |
| nvidia/nemotron-3.5-lightning | 1221 |
| google/gemma-4-31b-it | 1210 |
| cohere/north-mini-code | 1200 |
| baidu/ernie-5.5-flash-preview | 1150 |
| dots-studio/dots-3-note-preview | 1150 |
| google/gemma-4-26b-a4b-it | 1150 |
| inclusionai/ling-3.2-flash | 1150 |
| liquid/lfm-3-1.7b | 1150 |
| nvidia/nemotron-3-nano-omni-30b-a3b-reasoning | 1150 |
| particle-ai/particle-1 | 1150 |
| recursive/flint-v1.0 | 1150 |

No GPT, Claude, Gemini, or Llama model is in the catalog. The funded
public deployment caps routing at 31B parameters (Gemma 4 31B is the
largest) and drops the 550B outlier, and adds a local-execution mode so
the smallest models can run on community hardware.

## Determinism

Everything a request depends on is a function of the brief, the request,
and the gateway answers seen so far: no wall clock and no unseeded
randomness anywhere in a request path, so a replay makes exactly the live
pass's requests.

## Run it

```bash
# Node >= 22. Needs an OpenRouter key in OPENROUTER_API_KEY.
node main.ts
# The server listens on a port you choose and registers with the env
# (see the make.fun trial harness for the register/work protocol).
```

- `GET /v1/models` — the model list (OpenAI shape for
  `Authorization: Bearer`, Anthropic shape for `x-api-key`).
- `POST /v1/chat/completions` — OpenAI wire; `stream: true` for SSE.
- `POST /v1/messages` — Anthropic wire; `stream: true` for SSE.

## Status

Working prototype. Built for the make.fun `free-model-aggregator-02`
competition (November 2026), where it served real prompts end to end
through the live gateway. It has no public deployment and no users yet —
that is what the Mozilla Human-Scale AI Incubator proposal
(`../mozilla-incubator/proposal.md` in the author's workspace) funds:
a public deployment and a pilot with the community it is built for.

## License

MIT — see [LICENSE](LICENSE).
