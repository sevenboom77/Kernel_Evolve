# Host provider request limits

An optional process-static profile limit bounds provider dispatches for each session:

```json
{
  "limits": {
    "session.max_provider_requests": 24
  }
}
```

The value must be a positive safe integer. This is separate from a preset's per-turn `budget.max_steps`. The session allowance is shared across turns, model slots, compaction segments and retries, including adapter authentication recovery. Each attempt reserves a slot durably before network dispatch. Cancelled, failed or uncertain attempts keep their reservations; reopening the session retains the count and original cap. A lower configured cap tightens the original bound; raising or removing the setting does not restore allowance for an existing capped session.

Sessions with earlier provider work but no trusted admission history cannot establish their old physical request count. Applying a cap to such a legacy session exhausts its new allowance conservatively; use a fresh session for new work. Already persisted title costs still recover normally. Explicitly capped sessions use the prompt-based UI title fallback instead of spending a model request on an automatic title.

When the setting is omitted for a new session, existing behavior remains unchanged. A custom Provider must declare `supportsAttemptAdmission: true` and await the optional `beforeAttempt` callback before every physical dispatch. A false result or exception forbids that dispatch and terminates the invocation without retrying. Adapters or APIs that cannot enforce this boundary refuse capped inference before network access. The built-in Pi adapter supports observable HTTP APIs; bounded Codex requests use SSE and bounded HTTP rejects redirects. Opaque Google, Vertex, Bedrock and media dispatches are refused in capped sessions.

`x/core/provider-request-admitted` native events are durable reservation evidence and a conservative dispatch upper bound, including attempts that never reached the network. `request/sent` remains a receipt for a logical Provider invocation, and a compaction cost ledger can aggregate several attempts. Do not interpret either of those older counts as the physical dispatch count.
