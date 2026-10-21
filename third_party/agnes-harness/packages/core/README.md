# Core provider request admission

KernelOptions and SessionDeps accept the optional trusted `maxProviderRequests` positive safe integer. Host supplies it from [`session.max_provider_requests`](../host/README.md). It bounds the entire session independently of the preset's per-turn `maxSteps`.

The session's Provider wrapper attaches `beforeAttempt` to bounded inference. All SessionImpl objects sharing a SessionLog share an independent serialized admission queue; it does not reenter the phase lock. The existing log head compare-and-set and writer lease fence each durable append. Independent stale log writers cannot dispatch after a refused reservation.

The verified native ledger owns the budget history. System/trusted `x/core/provider-request-budget` anchors retain the cap; `x/core/provider-request-admitted` rows increase `used` once per admitted dispatch. Recovery validates session ownership, monotonic usage and a cap that never increases. No reservation is refunded after cancellation, failure or an uncertain process exit. Legacy provider work without these rows consumes a newly applied allowance conservatively.

Providers used with a cap must declare `supportsAttemptAdmission: true` and honor the callback before every physical attempt, including internal retries. An unsupported Provider or denied admission ends model work as a budget outcome without dispatch. New sessions without a cap retain the original Provider behavior. These ignorable evidence rows do not introduce a new protocol register or alter the primary program counter.
