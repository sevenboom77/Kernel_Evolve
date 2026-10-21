// Kernel workflow tools for the AGH harness-evolution project.
//
// Five tools wrap the frozen kernel_lab CLI (`python -m kernel_lab ...`) so the
// inner AGH session can search a GPU kernel end to end: read the contract,
// submit a candidate, poll the real job, read structured feedback. The plugin
// never evaluates anything itself: compilation, correctness and timing happen
// only in the independent evaluator process that consumes the same queue
// (docs/AGH_GPU_KERNEL_HARNESS_EVOLUTION_PLAN_20261003.md §7).
//
// Runtime configuration (which run, which interpreter) is read at call time
// from a JSON file written by the outer controller, so batches can switch runs
// without reinstalling the plugin and the inner agent cannot pick its own
// ledger: KERNEL_TOOLS_CONFIG, default
// /data/lizhijun/work/kernel-lab/harness-evolution/agh-adapter/kernel-tools.json
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

const Kind = Symbol.for('TypeBox.Kind')

const str = (extra = {}) => ({ [Kind]: 'String', type: 'string', ...extra })
const int = (extra = {}) => ({ [Kind]: 'Integer', type: 'integer', ...extra })
const obj = (properties, required = []) => ({
  [Kind]: 'Object',
  type: 'object',
  properties,
  required,
  additionalProperties: false,
})

const CONFIG_PATH =
  process.env.KERNEL_TOOLS_CONFIG ||
  '/data/lizhijun/work/kernel-lab/harness-evolution/agh-adapter/kernel-tools.json'

// One tool result the model sees is bounded anyway by the host's output guard;
// keeping the default transcript well under that keeps the summary readable.
const MAX_TEXT = 24000

function loadConfig() {
  try {
    const cfg = JSON.parse(readFileSync(CONFIG_PATH, 'utf8'))
    if (!cfg.kl_python || !cfg.kl_root || !cfg.run_id) {
      return { ok: false, reason: `config at ${CONFIG_PATH} must set kl_python, kl_root, run_id` }
    }
    return { ok: true, cfg }
  } catch (e) {
    return { ok: false, reason: `cannot read ${CONFIG_PATH}: ${e.message}` }
  }
}

function ok(structured, text) {
  // Scrub undefined: the wire shape is JSON, and a structured payload carrying
  // `undefined` values has been rejected by envelope validation in-session.
  const clean = JSON.parse(JSON.stringify(structured ?? {}))
  const body = text ?? JSON.stringify(clean, null, 1)
  return {
    content: [{ type: 'text', text: body.length > MAX_TEXT ? body.slice(0, MAX_TEXT) + '\n…truncated' : body }],
    structured: clean,
  }
}

function fail(message) {
  return { content: [{ type: 'text', text: message }], isError: true, structured: { error: message } }
}

function experimentBudget(payload) {
  const { cfg } = loadConfig()
  if (!payload?.budget || !Number.isInteger(cfg?.max_candidates)) return payload
  const used = payload.budget.source_candidates ?? 0
  const cap = Math.min(cfg.max_candidates, payload.budget.max_candidates ?? cfg.max_candidates)
  return { ...payload, budget: { ...payload.budget, max_candidates: cap, remaining: Math.max(0, cap - used) } }
}

const submissions = new Map()
async function serialSubmission(key, execute) {
  const previous = submissions.get(key) ?? Promise.resolve()
  const next = previous.catch(() => {}).then(execute)
  submissions.set(key, next)
  try { return await next } finally {
    if (submissions.get(key) === next) submissions.delete(key)
  }
}

async function klab(ctx, buildArgs, timeoutMs) {
  const { ok: cfgOk, cfg, reason } = loadConfig()
  if (!cfgOk) return { ok: false, error: reason, stdout: '', stderr: '' }
  const args = buildArgs(cfg)
  const taskIndex = args.indexOf('--task')
  if (cfg.workspaces && taskIndex >= 0 && cfg.workspaces[resolve(ctx.cwd)] !== args[taskIndex + 1]) {
    return { ok: false, error: 'task_id must match the registered task for this session workspace', stdout: '', stderr: '' }
  }
  const env = {
    PYTHONPATH: cfg.kl_root,
    XDG_CACHE_HOME: `${cfg.kl_root}/.cache`,
    TMPDIR: `${cfg.kl_root}/.cache/tmp`,
  }
  // The exec gate authorizes the *cwd* against the session workspace fence, so the
  // child starts in the session workspace (like the shell tool does). The interpreter
  // and the kernel_lab module are absolute paths: the tools are the fixed harness's
  // sanctioned bridge to the evaluator CLI, and the boundary that matters for the
  // experiment - candidate sources must live in the session workspace - is enforced
  // in kernel_submit below, not by where the process starts.
  const r = await ctx.exec([cfg.kl_python, '-m', 'kernel_lab', ...args], {
    cwd: ctx.cwd,
    env,
    timeoutMs,
  })
  let parsed = null
  try {
    parsed = JSON.parse(r.stdout)
  } catch {
    /* non-JSON stdout is reported raw */
  }
  return { ok: r.code === 0 && (parsed?.ok ?? true), parsed, stdout: r.stdout, stderr: r.stderr, code: r.code }
}

// ---------------------------------------------------------------- kernel_get_task
const getTaskTool = {
  inject: ['extension'],
  apply(ctxExt) {
    const agnes = ctxExt.extension()
    agnes.registerTool({
      name: 'kernel_get_task',
      description:
        'Read one kernel task contract: math definition, input domain, tolerance, dev/reserved shapes, ' +
        'budget, baseline qualification state, and the workspace where candidates must be written. ' +
        'Call this first. task_id is like K03.',
      parameters: obj({ task_id: str({ minLength: 2, maxLength: 16 }) }, ['task_id']),
      meta: {
        isReadOnly: true,
        isDestructive: false,
        isConcurrencySafe: true,
        isOpenWorld: false,
        replay: 'safe',
        costHint: undefined,
        deferLoading: false,
        requiresApproval: 'never',
      },
      async execute({ task_id }, ctx) {
        const r = await klab(ctx, (cfg) => ['dsh', 'digest', '--run', cfg.run_id, '--task', task_id])
        if (!r.parsed) return fail(r.error || `kernel_lab failed: ${r.stderr.slice(-500)}`)
        if (r.parsed.ok === false) return fail(JSON.stringify(r.parsed))
        // The model's own file tools stay fenced to the session workspace, so the
        // contract text must travel inside this tool's result: plugin code is
        // trusted in-process code and may read the frozen task package.
        let contract = null
        try {
          contract = readFileSync(r.parsed.task_md, 'utf8')
        } catch (e) {
          contract = `contract text unavailable: ${e.message}`
        }
        return ok({ ...experimentBudget(r.parsed), contract, workspace: ctx.cwd })
      },
    })
  },
}

// ---------------------------------------------------------------- kernel_submit
const submitTool = {
  inject: ['extension'],
  apply(ctxExt) {
    const agnes = ctxExt.extension()
    agnes.registerTool({
      name: 'kernel_submit',
      description:
        'Submit a candidate kernel for evaluation. candidate_dir is relative to the session workspace ' +
        'and must contain kernel.py defining ModelNew (see kernel_get_task). The source is frozen to a ' +
        'snapshot and queued; the independent evaluator compiles, checks numerics and times it on a real ' +
        'GPU. Returns a job receipt (job_id, candidate_id, source_hash) - poll kernel_get_job for the ' +
        'result. Submitting identical source twice just creates another candidate: write and check before ' +
        'you submit. Budget is enforced by the queue; a rejected submit is final for that task.',
      parameters: obj(
        {
          task_id: str({ minLength: 2, maxLength: 16 }),
          candidate_dir: str({ minLength: 1, maxLength: 512 }),
          parent_hash: str({ maxLength: 80 }),
          hypothesis: str({ maxLength: 2000 }),
        },
        ['task_id', 'candidate_dir'],
      ),
      meta: {
        isReadOnly: false,
        isDestructive: false,
        isConcurrencySafe: true,
        isOpenWorld: false,
        replay: 'never',
        costHint: { wallMs: 120000 },
        deferLoading: false,
        requiresApproval: 'never',
      },
      async execute({ task_id, candidate_dir, parent_hash, hypothesis }, ctx) {
        return serialSubmission(`${ctx.cwd}:${task_id}`, async () => {
        // The candidate must live inside this session's workspace: the tools
        // are the agent's write path into the run ledger, never a way to reach
        // another task's files or the harness itself.
        const abs = resolve(ctx.cwd, candidate_dir)
        if (!abs.startsWith(resolve(ctx.cwd) + '/') && abs !== resolve(ctx.cwd)) {
          return fail('candidate_dir must be inside the session workspace')
        }
        const { cfg, reason } = loadConfig()
        if (!cfg) return fail(reason)
        if (cfg.workspaces && cfg.workspaces[resolve(ctx.cwd)] !== task_id) {
          return fail('task_id must match the registered task for this session workspace')
        }
        if (Number.isFinite(cfg.deadline_ms) && Date.now() >= cfg.deadline_ms) {
          return fail('EXPERIMENT_DEADLINE: no new candidate may be submitted')
        }
        if (Number.isInteger(cfg.max_candidates)) {
          const state = await klab(ctx, (c) => ['task-status', '--run', c.run_id, '--task', task_id], 30000)
          if (!state.ok || !state.parsed?.budget) return fail(state.error || 'cannot verify remaining experiment budget')
          const used = state.parsed.budget.source_candidates
          if (!Number.isInteger(used)) return fail('cannot verify consumed experiment budget')
          if (used >= cfg.max_candidates) return fail(`BUDGET_EXHAUSTED: used=${used}, cap=${cfg.max_candidates}`)
        }
        const r = await klab(ctx, (cfg) => {
          const args = ['submit', '--run', cfg.run_id, '--task', task_id, '--candidate-dir', abs]
          if (parent_hash) args.push('--parent-hash', parent_hash)
          return args
        }, 30000)
        if (!r.parsed) return fail(r.error || `kernel_lab failed: ${r.stderr.slice(-500)}`)
        const out = { ...r.parsed, workspace: ctx.cwd, hypothesis: hypothesis ?? null }
        return r.parsed.ok === false
          ? { ...ok(out), isError: true }
          : ok(out)
        })
      },
    })
  },
}

// ---------------------------------------------------------------- kernel_get_job
function jobSummary(payload) {
  if (!payload) return payload
  const res = payload.result || {}
  return {
    job: payload.job,
    status: payload.status,
    has_result: payload.has_result,
    result: {
      status: res.status,
      phase: res.phase,
      correctness: res.correctness
        ? { ok: res.correctness.ok, passed: res.correctness.passed, total: res.correctness.total,
            first_failures: res.correctness.first_failures }
        : undefined,
      task_speedup: res.task_speedup,
      shape_speedups: res.shape_speedups,
      promoted: res.promoted,
      promotion_reason: res.promotion_reason,
      error: res.error,
      contaminated: res.contaminated,
      contamination_note: res.contamination_note,
      below_measurement_floor: res.below_measurement_floor,
      measurement_floor_note: res.measurement_floor_note,
      launch_bound: res.launch_bound,
      launch_bound_note: res.launch_bound_note,
      baseline_status: res.baseline_status,
      seconds: res.seconds,
    },
  }
}

const getJobTool = {
  inject: ['extension'],
  apply(ctxExt) {
    const agnes = ctxExt.extension()
    agnes.registerTool({
      name: 'kernel_get_job',
      description:
        'Query one kernel job by id. Returns the real job stage and numerical results: status, ' +
        'correctness counts, per-shape speedups versus the strongest qualified baseline of the same run, ' +
        'promotion decision, and measurement notes (cache floor / launch-bound / interference). ' +
        'Set wait_s to poll the queue inside this call (bounded) instead of returning "queued" at once.',
      parameters: obj(
        {
          job_id: str({ minLength: 4, maxLength: 64 }),
          wait_s: int({ minimum: 0, maximum: 90 }),
        },
        ['job_id'],
      ),
      meta: {
        isReadOnly: true,
        isDestructive: false,
        isConcurrencySafe: true,
        isOpenWorld: false,
        replay: 'safe',
        costHint: { wallMs: 250000 },
        deferLoading: false,
        requiresApproval: 'never',
      },
      async execute({ job_id, wait_s }, ctx) {
        // Stay strictly inside the tool's own soft deadline: a call the kernel
        // has to cut off comes back as a timeout error with no data at all,
        // while returning early still gives the model the real job state.
        const budgetMs = Math.min((wait_s ?? 0) * 1000, Math.max(0, ctx.timeoutMs - 20_000))
        const deadline = Date.now() + budgetMs
        for (;;) {
          const r = await klab(ctx, (cfg) => ['job', '--run', cfg.run_id, '--id', job_id])
          if (!r.parsed) return fail(r.error || `kernel_lab failed: ${r.stderr.slice(-500)}`)
          const status = r.parsed.status || r.parsed.job?.status
          if (!wait_s || (status && status !== 'queued' && status !== 'running')) {
            return ok(jobSummary(r.parsed))
          }
          const remaining = deadline - Date.now()
          if (remaining < 8_000) return ok({ ...jobSummary(r.parsed), waited_s: budgetMs / 1000 })
          await new Promise((done) => setTimeout(done, Math.min(5000, remaining)))
        }
      },
    })
  },
}

// ---------------------------------------------------------------- kernel_get_task_status
const taskStatusTool = {
  inject: ['extension'],
  apply(ctxExt) {
    const agnes = ctxExt.extension()
    agnes.registerTool({
      name: 'kernel_get_task_status',
      description:
        'Task-level progress: current champion and its task speedup, remaining candidate budget, the ' +
        'consecutive non-improving count and the stagnation stop condition, qualified baselines and the ' +
        'strongest baseline per shape. Use it to decide whether to keep iterating or stop.',
      parameters: obj({ task_id: str({ minLength: 2, maxLength: 16 }) }, ['task_id']),
      meta: {
        isReadOnly: true,
        isDestructive: false,
        isConcurrencySafe: true,
        isOpenWorld: false,
        replay: 'safe',
        costHint: undefined,
        deferLoading: false,
        requiresApproval: 'never',
      },
      async execute({ task_id }, ctx) {
        const r = await klab(ctx, (cfg) => ['task-status', '--run', cfg.run_id, '--task', task_id])
        if (!r.parsed) return fail(r.error || `kernel_lab failed: ${r.stderr.slice(-500)}`)
        return ok(experimentBudget(r.parsed))
      },
    })
  },
}

// ---------------------------------------------------------------- kernel_read_feedback
const readFeedbackTool = {
  inject: ['extension'],
  apply(ctxExt) {
    const agnes = ctxExt.extension()
    agnes.registerTool({
      name: 'kernel_read_feedback',
      description:
        'Structured feedback for a finished job, sliced for action: detail="summary" gives the verdict ' +
        'and next-step classification (compile error vs numeric mismatch vs slow vs no-gain), "errors" ' +
        'gives failure traces, "timing" gives the per-shape latency table with measurement notes, "full" ' +
        'gives everything the evaluator recorded. Numbers are exactly the evaluator\'s numbers.',
      parameters: obj(
        {
          job_id: str({ minLength: 4, maxLength: 64 }),
          detail: { [Kind]: 'String', type: 'string', enum: ['summary', 'errors', 'timing', 'full'] },
        },
        ['job_id'],
      ),
      meta: {
        isReadOnly: true,
        isDestructive: false,
        isConcurrencySafe: true,
        isOpenWorld: false,
        replay: 'safe',
        costHint: undefined,
        deferLoading: false,
        requiresApproval: 'never',
      },
      async execute({ job_id, detail }, ctx) {
        const r = await klab(ctx, (cfg) => ['job', '--run', cfg.run_id, '--id', job_id])
        if (!r.parsed) return fail(r.error || `kernel_lab failed: ${r.stderr.slice(-500)}`)
        const p = r.parsed
        const res = p.result || {}
        const kind = detail || 'summary'
        if (kind === 'full') return ok(p)

        const classification =
          res.status === 'complete'
            ? res.promoted
              ? 'promoted: better than the incumbent champion'
              : 'correct but no gain: ' + (res.promotion_reason || 'slower than the strongest baseline')
            : res.status === 'incorrect'
              ? 'numeric mismatch: fix correctness before chasing speed'
              : res.status === 'compile_fail'
                ? 'compile error: fix the build before chasing speed'
                : res.status === 'timeout'
                  ? 'timeout: the candidate is too slow to finish in the job budget'
                  : res.status === 'oom'
                    ? 'out of memory: reduce workspace or buffers'
                    : res.status === 'env_error' || res.status === 'crashed'
                      ? 'environment/evaluator problem, not your kernel: retry once, then report'
                      : `status ${res.status}`

        if (kind === 'errors') {
          return ok({
            job_id,
            status: res.status,
            classification,
            error: res.error,
            traceback: res.traceback,
            correctness: res.correctness,
            child_errors: res.errors,
          })
        }
        if (kind === 'timing') {
          return ok({
            job_id,
            status: res.status,
            classification,
            task_speedup: res.task_speedup,
            shape_speedups: res.shape_speedups,
            candidate_us: res.candidate_us,
            strongest_baseline: res.strongest_baseline,
            unpaired_baseline_reference: res.unpaired_baseline_reference,
            below_measurement_floor: res.below_measurement_floor,
            measurement_floor_note: res.measurement_floor_note,
            launch_bound: res.launch_bound,
            launch_bound_note: res.launch_bound_note,
            contaminated: res.contaminated,
            contamination_note: res.contamination_note,
          })
        }
        return ok({
          job_id,
          status: res.status,
          classification,
          correctness: res.correctness
            ? { ok: res.correctness.ok, passed: res.correctness.passed, total: res.correctness.total,
                first_failures: res.correctness.first_failures }
            : undefined,
          task_speedup: res.task_speedup,
          shape_speedups: res.shape_speedups,
          promoted: res.promoted,
          promotion_reason: res.promotion_reason,
          error: res.error,
          notes: [res.measurement_floor_note, res.launch_bound_note, res.contamination_note]
            .filter(Boolean),
        })
      },
    })
  },
}

export const kernelWorkflowTools = {
  inject: ['extension'],
  apply(ctx) {
    getTaskTool.apply(ctx)
    submitTool.apply(ctx)
    getJobTool.apply(ctx)
    taskStatusTool.apply(ctx)
    readFeedbackTool.apply(ctx)
    ctx.extension().on('session_start', () => {
      ctx.extension().ctx.log.info('kernel_* workflow tools registered')
    })
  },
}

// Fixed H0 preparation: these limits are identical for every compared version.
export const presets = {
  'kernel-search': {
    name: 'kernel-search', extends: 'standard',
    model: { max_tokens: 4096, retry: { max_attempts: 1 } },
    budget: { max_steps: 24, on_exceed: 'deny' },
    subagent: { max_depth: 0, max_fan_out: 0 },
  },
}
