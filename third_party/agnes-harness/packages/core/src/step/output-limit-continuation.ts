import type { Event } from '../types.js'

/** One recovery per turn, within the durable physical request cap and actual tool disclosure. */
export function outputLimitContinuation(input: {
  slot: string
  offered: ReadonlySet<string>
  allowance: { cap: number; used: number; remaining: number } | undefined
  aborted: boolean
  scope: string
  history: readonly Event[]
  lane: string
  fromSeq: number
}) {
  if (
    input.slot !== 'primary' ||
    input.aborted ||
    !input.allowance ||
    input.allowance.remaining <= 0 ||
    !/^[a-f0-9]{64}$/.test(input.scope) ||
    !['kernel_get_task', 'kernel_submit', 'kernel_get_job'].every((name) => input.offered.has(name))
  )
    return undefined
  // A trusted marker with a changed registry/scope or malformed payload still consumes recovery.
  // Untrusted lookalikes and other lanes cannot change this turn's decision.
  if (
    input.history.some(
      (row) =>
        row.type === 'x/core/output-limit-continuation' &&
        row.origin === 'system' &&
        row.trust === 'trusted' &&
        row.lane === input.lane &&
        row.seq >= input.fromSeq,
    )
  )
    return undefined
  return { schemaVersion: 1, scope: input.scope }
}
