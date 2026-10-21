import type { ToolContext, ToolResult } from '@agnes/extension-api'
import type { ResolvedToolCallPolicy } from '@agnes/protocol'
import { isToolResult } from '../effects/tool-dispatch.js'

const record = (value: unknown): Record<string, unknown> | undefined =>
  value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined

/** Continue one explicit safe wait inside its original deadline, retaining the first observation. */
export async function boundedPendingJobQuery(input: {
  name: string
  args: unknown
  policy: ResolvedToolCallPolicy
  context: ToolContext
  allowContinuation: boolean
  execute(context: ToolContext): Promise<ToolResult>
  recordPending(result: ToolResult): Promise<unknown>
}): Promise<ToolResult> {
  const started = performance.now()
  const first = await input.execute(input.context)
  if (
    !input.allowContinuation ||
    input.name !== 'kernel_get_job' ||
    input.policy.isReadOnly !== true ||
    input.policy.isDestructive !== false ||
    input.policy.replay !== 'safe' ||
    !isToolResult(first)
  )
    return first
  const args = record(input.args)
  const data = record(first.structured)
  const job = record(data?.job)
  const remaining = () => Math.max(0, input.context.timeoutMs - Math.ceil(performance.now() - started))
  if (
    first.isError ||
    first.terminate ||
    Object.hasOwn(first, 'deferred') ||
    typeof args?.wait_s !== 'number' ||
    !Number.isSafeInteger(args.wait_s) ||
    args.wait_s < 1 ||
    args.wait_s > 90 ||
    typeof args.job_id !== 'string' ||
    args.job_id.length < 4 ||
    args.job_id.length > 64 ||
    data?.has_result !== false ||
    data.status !== 'running' ||
    job?.status !== data.status ||
    job.kind !== 'candidate' ||
    job.phase !== 'dev' ||
    job.job_id !== args.job_id ||
    typeof job.source_hash !== 'string' ||
    !/^[a-f0-9]{64}$/.test(job.source_hash) ||
    input.context.signal.aborted ||
    !Number.isFinite(remaining()) ||
    remaining() < 30_000
  )
    return first
  try {
    await input.recordPending(first)
  } catch {
    // Never make the extra query if its original pending observation cannot be retained.
    return first
  }
  const timeoutMs = remaining()
  if (input.context.signal.aborted || timeoutMs < 30_000) return first
  return input.execute({ ...input.context, timeoutMs })
}
