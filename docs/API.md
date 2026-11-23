# API

All chat endpoints are OpenAI-compatible; point the SDK's `base_url` at `http://localhost:8000/v1`.
In lite mode (`REQUIRE_AUTH=false`) any bearer token is accepted. Errors use `{"error": {"message","type","code"}}`.

## `POST /v1/chat/completions`
Standard OpenAI body. `model` may be any model id/alias in `models.yaml`, a vendor name (`gpt-4o`, `llama-3.3-70b-versatile`, `gemini-2.5-flash`) or `"auto"`. Unsupported: `n>1`. Optional `x_switchboard` object carries test hints for the mock provider and is never forwarded upstream.

Request headers: `X-Cache: bypass`, `X-Request-Id`. Responses add:

| Header | Meaning |
|---|---|
| `X-Cache` | `HIT-EXACT` / `HIT-SEMANTIC` / `MISS` |
| `X-Cache-Similarity` | cosine similarity (semantic hits) |
| `X-Provider`, `X-Model` | what actually served the request |
| `X-Routing-Tier` | tier chosen (cheap/balanced/strong) |
| `X-Cost-USD` | actually billed (0 on free tiers) |
| `X-Cost-Equivalent-USD` | cost at the reference price |
| `X-Cost-Saved-USD` | (same tokens on the strongest model, no cache) minus equivalent cost |
| `X-Budget-Warning` | `soft` or `downgraded` |

Cost headers are omitted on streams. Cache applies when `temperature` is set and `<= 0.3`.

## `GET /v1/models`
Lists `auto` plus every enabled model with tier and context window.

## Batch jobs
* `POST /v1/jobs` - JSON `{"requests":[chat bodies], "priority":"interactive|normal|batch|bulk", "model":"auto|<id>", "max_concurrency":8, "idempotency_key":"...", "callback_url":"https://...", "metadata":{}}`; or `Content-Type: application/x-ndjson` with one chat body per line (options as query params; `Idempotency-Key` header). Returns `202` (or `200` for a repeated idempotency key).
* `GET /v1/jobs/{id}` - status, counts, cost so far (billed/equivalent/saved/cache hits), ETA.
* `GET /v1/jobs/{id}/results` - streaming JSONL, one `{"index","status","response","error","attempts"}` per finished task.
* `POST /v1/jobs/{id}/cancel`, `GET /v1/jobs`.

Run workers with `switchboard worker` (any number of processes sharing the DB), or `switchboard serve --with-worker`.

## Admin (`X-Admin-Key: $ADMIN_KEY`)
`GET /admin/usage?group_by=model|key|day|provider|path|cache`, `GET /admin/savings`, `GET /admin/stats` (dashboard feed), `POST|GET /admin/keys`, `POST /admin/cache/invalidate {"namespace"|"model_family"}`, `GET /admin/dlq`, `POST /admin/dlq/requeue`, `GET /admin/routing`, `GET /admin/breakers`.

## Ops
`GET /health`, `GET /ready`, `GET /metrics` (Prometheus), `GET /dashboard`.
