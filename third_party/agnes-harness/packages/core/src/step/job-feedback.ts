import type { ToolResult } from '@agnes/extension-api'

const failedStatuses = new Set(['compile_fail', 'incorrect', 'timeout', 'oom', 'crashed'])
const recoveryText =
  'The queried job has finished with a failure. A successful status query does not mean the computation succeeded. First inspect its diagnostics and repair the source or environment before retrying unchanged input.'
const embeddedDiagnosticText =
  'The queried job has finished with a failure. The error diagnostic is already included here. Repair the source or environment using that diagnostic; read an extra diagnostic artifact only if the included error is insufficient.'
const compilerLocationText =
  "The queried job has finished with a failure. The included compiler excerpt identifies the source location but omits the underlying cause. Inspect this job's full error or traceback diagnostics before revising the source or resubmitting. Keep the reference and tolerances unchanged."
const compilerTracebackText =
  'The queried job has finished with a failure. The underlying compiler cause is already included in the traceback. Use the exception type and message in that traceback to repair the source before resubmitting. Keep the reference and tolerances unchanged; read an extra diagnostic artifact only if the included traceback is insufficient.'

function record(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined
}

function numericalRecoveryText(outcome: Record<string, unknown>): string | undefined {
  const correctness = record(outcome.correctness)
  const passed = correctness?.passed
  const total = correctness?.total
  if (
    outcome.status !== 'incorrect' ||
    correctness?.ok !== false ||
    typeof passed !== 'number' ||
    typeof total !== 'number' ||
    !Number.isSafeInteger(passed) ||
    !Number.isSafeInteger(total) ||
    passed < 0 ||
    passed >= total ||
    !Array.isArray(correctness.first_failures) ||
    !correctness.first_failures.some((failure) => {
      const problems = record(failure)?.problems
      return Array.isArray(problems) && problems.some((p) => typeof p === 'string' && p.trim().length > 0)
    })
  )
    return undefined
  return `The queried job has finished with a failure: ${passed} of ${total} correctness cases passed. The numerical failure details are already included here. Use the failing cases and output mismatch details to repair the failing computation before speed optimization or resubmission. Keep the reference and tolerances unchanged; read an extra diagnostic artifact only if the included details are insufficient.`
}

/** Explain an explicit failed job without changing the query's result or error status. */
export function withTerminalJobFeedback(result: ToolResult): ToolResult {
  if (Object.hasOwn(result, 'deferred')) return result
  const data = record(result.structured)
  const job = record(data?.job)
  const outcome = record(data?.result)
  if (
    !data ||
    !job ||
    !outcome ||
    data.has_result !== true ||
    typeof data.status !== 'string' ||
    !failedStatuses.has(data.status) ||
    job.status !== data.status ||
    outcome.status !== data.status ||
    typeof job.job_id !== 'string' ||
    job.job_id.length === 0 ||
    job.job_id.length > 128 ||
    typeof job.source_hash !== 'string' ||
    !/^[a-f0-9]{64}$/.test(job.source_hash) ||
    (outcome.source_hash !== undefined && outcome.source_hash !== job.source_hash) ||
    (outcome.job_id !== undefined && outcome.job_id !== job.job_id)
  )
    return result
  const embeddedDiagnostic =
    typeof outcome.error === 'string' &&
    /(?:Traceback \(most recent call last\):|(?:[A-Za-z_][A-Za-z0-9_]*Error|Exception):)/.test(outcome.error)
  const compilerLocationOnly =
    data.status === 'compile_fail' &&
    typeof outcome.error === 'string' &&
    outcome.error.length <= 32768 &&
    (outcome.phase === undefined || outcome.phase === job.phase) &&
    /^CompilationError:\s*at \d+:\d+:[ \t]*\r?\n/.test(outcome.error) &&
    /\r?\n[ \t]*[~^]+[ \t]*(?:\r?\n[ \t]*)*$/.test(outcome.error)
  const includedCompilerCause =
    compilerLocationOnly &&
    typeof outcome.traceback === 'string' &&
    outcome.traceback.length <= 32768 &&
    /^Traceback \(most recent call last\):\r?\n/.test(outcome.traceback) &&
    /(?:^|\r?\n)(?!(?:[A-Za-z_]\w*\.)*CompilationError:)(?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*(?:Error|Exception):[ \t]*\S[^\r\n]*/.test(
      outcome.traceback,
    )
  const guidance =
    numericalRecoveryText(outcome) ??
    (includedCompilerCause
      ? compilerTracebackText
      : compilerLocationOnly
        ? compilerLocationText
        : embeddedDiagnostic
          ? embeddedDiagnosticText
          : recoveryText)
  return {
    ...result,
    content: [...result.content, { type: 'text', text: guidance }],
  }
}
