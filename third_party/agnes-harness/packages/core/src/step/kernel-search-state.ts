import { isToolResult } from '../effects/tool-dispatch.js'
import {
  hasAuthenticToolPolicyHash,
  hasCompleteToolPolicyEnvelope,
  hasTrustedToolCallProvenance,
  type PersistedToolPolicyFields,
} from '../registry/tool-policy.js'
import { canonicalJson, sha256Hex } from '../request/hash.js'
import type { Event } from '../types.js'
import {
  isNumericalFailureClass,
  type NumericalFailureClass,
  numericalFailureClass,
  numericalFailureFingerprint,
  numericRepairFocus,
} from './numerical-failure.js'

export type KernelSearchObservation = {
  schemaVersion: 1
  scope: string
  run_id: string
  task_id: string
  job_id: string
  source_hash: string
  status: 'compile_fail' | 'incorrect' | 'complete' | 'crashed' | 'timeout'
  passed?: number
  total?: number
  numerical_failure_class?: NumericalFailureClass
  numerical_failure_fingerprint?: string
  observed_dev_task_speedup?: number
  below_measurement_floor?: boolean
  launch_bound?: boolean
}

type KernelSearchSubmission = {
  schemaVersion: 1
  scope: string
  task_id: string
  job_id: string
  candidate_id: string
  source_hash: string
  call_seq: number
}

function record(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined
}

function identifier(value: unknown, max = 128): value is string {
  return (
    typeof value === 'string' && value.length > 0 && value.length <= max && /^[A-Za-z0-9_.:-]+$/.test(value)
  )
}

function count(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0
}

const hash = (value: unknown): value is string => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value)

/** The scope survives replay but never crosses a session, lane, workspace, turn or registry. */
export function kernelSearchScope(
  sessionKey: string,
  lane: string,
  cwd: string,
  turn: number,
  snapshotHash: string,
): string {
  if (!sessionKey || !lane || !cwd || !count(turn) || turn === 0 || !hash(snapshotHash)) return ''
  return sha256Hex(canonicalJson({ sessionKey, lane, cwd, turn, snapshotHash }))
}

function authenticatedCall(call: Event, name: string) {
  const data = record(call.data)
  const envelope = data as PersistedToolPolicyFields | undefined
  if (
    !hasTrustedToolCallProvenance(call) ||
    !data ||
    data.name !== name ||
    !identifier(data.toolUseId) ||
    !count(call.seq) ||
    call.seq === 0 ||
    !hasCompleteToolPolicyEnvelope(envelope) ||
    !hasAuthenticToolPolicyHash(envelope) ||
    !hash(envelope.definitionFingerprint) ||
    envelope.executionDomain !== 'workspace' ||
    envelope.resolvedPolicy.isOpenWorld !== false ||
    envelope.resolvedPolicy.isDestructive !== false
  )
    return undefined
  const args = record(data.args)
  return args ? { data, args, policy: envelope.resolvedPolicy } : undefined
}

function submission(value: unknown, scope: string): KernelSearchSubmission | undefined {
  const data = record(value)
  if (
    data?.schemaVersion !== 1 ||
    data.scope !== scope ||
    !hash(scope) ||
    !identifier(data.task_id, 16) ||
    !identifier(data.job_id) ||
    !identifier(data.candidate_id) ||
    !hash(data.source_hash) ||
    !count(data.call_seq) ||
    data.call_seq === 0
  )
    return undefined
  return {
    schemaVersion: 1,
    scope,
    task_id: data.task_id,
    job_id: data.job_id,
    candidate_id: data.candidate_id,
    source_hash: data.source_hash,
    call_seq: data.call_seq,
  }
}

/** Record only the real successful receipt, before any extension result replacement. */
export function observeKernelSubmission(input: { call: Event; result: unknown; scope: string; cwd: string }) {
  const call = authenticatedCall(input.call, 'kernel_submit')
  if (
    call?.policy.isReadOnly !== false ||
    call.policy.replay !== 'never' ||
    !isToolResult(input.result) ||
    input.result.isError ||
    input.result.terminate ||
    Object.hasOwn(input.result, 'deferred')
  )
    return undefined
  const receipt = record(input.result.structured)
  if (receipt?.ok !== true || receipt.phase !== 'dev' || receipt.workspace !== input.cwd) return undefined
  return submission(
    {
      schemaVersion: 1,
      scope: input.scope,
      task_id: call.args.task_id,
      job_id: receipt.job_id,
      candidate_id: receipt.candidate_id,
      source_hash: receipt.source_hash,
      call_seq: input.call.seq,
    },
    input.scope,
  )
}

/** Bind a terminal query to its authentic, current-turn submission and durable receipt. */
export function observeKernelSearch(input: {
  call: Event
  result: unknown
  scope: string
  history: readonly Event[]
  lane: string
  fromSeq: number
}): KernelSearchObservation | undefined {
  const call = authenticatedCall(input.call, 'kernel_get_job')
  if (
    call?.policy.isReadOnly !== true ||
    call.policy.replay !== 'safe' ||
    input.call.lane !== input.lane ||
    input.call.seq < input.fromSeq ||
    !count(input.fromSeq) ||
    !hash(input.scope) ||
    !isToolResult(input.result) ||
    input.result.isError ||
    input.result.terminate ||
    Object.hasOwn(input.result, 'deferred')
  )
    return undefined
  const data = record(input.result.structured)
  const job = record(data?.job)
  const outcome = record(data?.result)
  if (
    !data ||
    !job ||
    !outcome ||
    data.has_result !== true ||
    job.kind !== 'candidate' ||
    job.phase !== 'dev' ||
    outcome.phase !== 'dev' ||
    outcome.contaminated !== false ||
    job.status !== data.status ||
    outcome.status !== data.status ||
    !identifier(job.run_id) ||
    !identifier(job.task_id, 16) ||
    !identifier(job.candidate_id) ||
    job.job_id !== call.args.job_id ||
    !identifier(job.job_id) ||
    !hash(job.source_hash) ||
    !['compile_fail', 'incorrect', 'complete', 'crashed', 'timeout'].includes(String(data.status))
  )
    return undefined
  for (const key of ['run_id', 'task_id', 'candidate_id', 'job_id', 'source_hash']) {
    if (outcome[key] !== undefined && outcome[key] !== job[key]) return undefined
  }
  const rows = input.history.filter(
    (row) =>
      row.lane === input.lane &&
      row.trust === 'trusted' &&
      row.seq >= input.fromSeq &&
      row.seq < input.call.seq,
  )
  const bound = rows.some((row) => {
    if (row.type !== 'x/core/kernel-search-submission' || row.origin !== 'system') return false
    const accepted = submission(row.data, input.scope)
    if (
      !accepted ||
      accepted.job_id !== job.job_id ||
      accepted.task_id !== job.task_id ||
      accepted.candidate_id !== job.candidate_id ||
      accepted.source_hash !== job.source_hash ||
      row.sourceEventSeqs?.length !== 1 ||
      row.sourceEventSeqs[0] !== accepted.call_seq
    )
      return false
    const submitCall = rows.find((source) => source.seq === accepted.call_seq)
    const authentic = submitCall && authenticatedCall(submitCall, 'kernel_submit')
    if (
      authentic?.policy.isReadOnly !== false ||
      authentic.policy.replay !== 'never' ||
      authentic.args.task_id !== job.task_id
    )
      return false
    return rows.some((receiptRow) => {
      const receiptData = record(receiptRow.data)
      const receipt = record(receiptData?.structured)
      return (
        receiptRow.type === 'tool/result' &&
        receiptRow.origin === 'tool:kernel_submit' &&
        receiptRow.seq > accepted.call_seq &&
        receiptRow.seq < row.seq &&
        receiptRow.sourceEventSeqs?.length === 1 &&
        receiptRow.sourceEventSeqs[0] === accepted.call_seq &&
        receiptData?.toolUseId === authentic.data.toolUseId &&
        receiptData?.isError === false &&
        receipt?.ok === true &&
        receipt.phase === 'dev' &&
        receipt.job_id === accepted.job_id &&
        receipt.candidate_id === accepted.candidate_id &&
        receipt.source_hash === accepted.source_hash
      )
    })
  })
  if (!bound) return undefined
  const correctness = record(outcome.correctness)
  if (
    (data.status === 'complete' || data.status === 'incorrect') &&
    correctness?.ok !== (data.status === 'complete')
  )
    return undefined
  return observation(
    {
      schemaVersion: 1,
      scope: input.scope,
      run_id: job.run_id,
      task_id: job.task_id,
      job_id: job.job_id,
      source_hash: job.source_hash,
      status: data.status,
      ...(correctness ? { passed: correctness.passed, total: correctness.total } : {}),
      numerical_failure_class: data.status === 'incorrect' ? numericalFailureClass(correctness) : undefined,
      numerical_failure_fingerprint:
        data.status === 'incorrect' ? numericalFailureFingerprint(correctness) : undefined,
      observed_dev_task_speedup: outcome.task_speedup,
      below_measurement_floor: outcome.below_measurement_floor,
      launch_bound: outcome.launch_bound,
    },
    input.scope,
  )
}

function observation(value: unknown, scope: string): KernelSearchObservation | undefined {
  const data = record(value)
  if (
    data?.schemaVersion !== 1 ||
    data.scope !== scope ||
    !identifier(data.run_id) ||
    !identifier(data.task_id, 16) ||
    !identifier(data.job_id) ||
    typeof data.source_hash !== 'string' ||
    !/^[a-f0-9]{64}$/.test(data.source_hash) ||
    !['compile_fail', 'incorrect', 'complete', 'crashed', 'timeout'].includes(String(data.status))
  )
    return undefined
  const status = data.status as KernelSearchObservation['status']
  const validCounts = count(data.passed) && count(data.total) && data.total > 0 && data.passed <= data.total
  if (
    (status === 'complete' || status === 'incorrect') &&
    (!validCounts || (status === 'complete' ? data.passed !== data.total : data.passed === data.total))
  )
    return undefined
  const measured =
    status === 'complete' &&
    typeof data.observed_dev_task_speedup === 'number' &&
    Number.isFinite(data.observed_dev_task_speedup) &&
    data.observed_dev_task_speedup > 0
  return {
    schemaVersion: 1,
    scope,
    run_id: data.run_id,
    task_id: data.task_id,
    job_id: data.job_id,
    source_hash: data.source_hash,
    status,
    ...(validCounts ? { passed: data.passed as number, total: data.total as number } : {}),
    ...(status === 'incorrect' && isNumericalFailureClass(data.numerical_failure_class)
      ? { numerical_failure_class: data.numerical_failure_class }
      : {}),
    ...(status === 'incorrect' && hash(data.numerical_failure_fingerprint)
      ? { numerical_failure_fingerprint: data.numerical_failure_fingerprint }
      : {}),
    ...(measured
      ? {
          observed_dev_task_speedup: data.observed_dev_task_speedup as number,
          ...(typeof data.below_measurement_floor === 'boolean'
            ? { below_measurement_floor: data.below_measurement_floor }
            : {}),
          ...(typeof data.launch_bound === 'boolean' ? { launch_bound: data.launch_bound } : {}),
        }
      : {}),
  }
}

/** Rebuild a bounded same-scope state from durable observations, including after compaction. */
export function kernelSearchNote(scope: string, history: readonly Event[], lane: string): string {
  if (!/^[a-f0-9]{64}$/.test(scope)) return ''
  const rows = history
    .filter(
      (row) =>
        row.type === 'x/core/kernel-search-observation' &&
        row.lane === lane &&
        row.origin === 'system' &&
        row.trust === 'trusted' &&
        Number.isSafeInteger(row.seq) &&
        row.seq > 0,
    )
    .sort((a, b) => a.seq - b.seq)
    .flatMap((row) => {
      const parsed = observation(row.data, scope)
      return parsed ? [parsed] : []
    })
  const latest = rows.at(-1)
  if (!latest) return ''
  const sources = new Map<string, KernelSearchObservation>()
  const jobs = new Map<string, string>()
  for (const row of rows) {
    if (row.run_id !== latest.run_id || row.task_id !== latest.task_id) continue
    const known = jobs.get(row.job_id)
    if (known && known !== row.source_hash) continue
    jobs.set(row.job_id, row.source_hash)
    sources.delete(row.source_hash)
    sources.set(row.source_hash, row)
  }
  const accepted = [...sources.values()]
  const lastAccepted = accepted.at(-1)
  if (!lastAccepted) return ''
  let best: KernelSearchObservation | undefined
  for (const row of accepted) {
    if (
      row.observed_dev_task_speedup !== undefined &&
      row.observed_dev_task_speedup >= (best?.observed_dev_task_speedup ?? 0)
    )
      best = row
  }
  const facts = accepted.map(
    ({ schemaVersion: _schema, scope: _scope, run_id: _run, task_id: _task, ...fact }) => fact,
  )
  const trim = () => {
    const index = facts.findIndex(
      (fact) => fact.source_hash !== best?.source_hash && fact.source_hash !== lastAccepted.source_hash,
    )
    if (index < 0) return false
    facts.splice(index, 1)
    return true
  }
  while (facts.length > 4) trim()
  const category = lastAccepted.numerical_failure_class
  const repeatedDiagnostic =
    lastAccepted.status === 'incorrect' &&
    lastAccepted.numerical_failure_fingerprint &&
    accepted.some(
      (row) =>
        row.status === 'incorrect' &&
        row.source_hash !== lastAccepted.source_hash &&
        row.numerical_failure_fingerprint === lastAccepted.numerical_failure_fingerprint,
    )
  const repeatedClass =
    lastAccepted.status === 'incorrect' &&
    category &&
    accepted.filter((row) => row.status === 'incorrect' && row.numerical_failure_class === category).length >=
      2
      ? category
      : undefined
  const next =
    lastAccepted.status === 'compile_fail'
      ? 'inspect_compile_diagnostic'
      : lastAccepted.status === 'incorrect'
        ? repeatedDiagnostic
          ? 'change_numeric_repair_strategy_for_repeated_DEV_diagnostic'
          : repeatedClass
            ? 'change_numeric_repair_strategy_for_repeated_DEV_failure_class'
            : 'repair_numerics_before_speed_changes'
        : lastAccepted.status === 'crashed'
          ? 'inspect_runtime_failure_before_speed_changes'
          : lastAccepted.status === 'timeout'
            ? 'inspect_timeout_before_resubmitting'
            : lastAccepted.observed_dev_task_speedup !== undefined && best !== lastAccepted
              ? 'compare_prior_correct_DEV_source_before_further_tuning'
              : (lastAccepted.observed_dev_task_speedup ?? 1) < 1
                ? 'change_strategy_for_correct_but_slow_DEV_source'
                : 'optimize_correct_DEV_source'
  const encode = () =>
    'These are observed DEV outcomes for this search, not final acceptance. ' +
    'Preserve measurement warnings when comparing observed speeds. ' +
    JSON.stringify({
      run_id: latest.run_id,
      task_id: latest.task_id,
      sources: facts,
      ...(best
        ? {
            best_observed_correct_DEV_source_index: facts.findIndex(
              (fact) => fact.source_hash === best.source_hash,
            ),
            ...(best !== lastAccepted
              ? { comparison_to_best_observed_correct_DEV_source: 'prior_correct_DEV_source_available' }
              : {}),
          }
        : {}),
      next_action_for_latest_observed_source: next,
      ...(repeatedDiagnostic
        ? {
            repair_focus:
              'Different source hashes yielded the same bounded DEV mismatch diagnostics. Recheck reference indexing, masking, reduction axes and intermediate casts. Changing blocks or warps alone has not removed this failure.' +
              (category ? ` ${numericRepairFocus[category]}` : ''),
          }
        : category
          ? { repair_focus: numericRepairFocus[category] }
          : {}),
    }) +
    ' Keep reference and tolerances unchanged; assess each source by its full hash.'
  while (new TextEncoder().encode(encode()).length > 2048) {
    if (!trim()) return ''
  }
  return facts.length ? encode() : ''
}
