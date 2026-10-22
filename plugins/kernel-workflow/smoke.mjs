// No-model verification of the kernel_* tools (backend.md: load the module,
// capture registered definitions, exercise execute() against a real klab run).
//   node plugins/kernel-workflow/smoke.mjs
import { execFileSync } from 'node:child_process'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { kernelWorkflowTools } from './index.mjs'

const registered = {}
const execCalls = []
const fakeCtx = {
  cwd: mkdtempSync(join(tmpdir(), 'kw-smoke-')),
  timeoutMs: 60_000,
  exec: (argv, opts = {}) =>
    new Promise((done) => {
      execCalls.push({ argv: argv.slice(0, 2), cwd: opts.cwd })
      try {
        const stdout = execFileSync(argv[0], argv.slice(1), {
          cwd: opts.cwd,
          env: { ...process.env, ...(opts.env ?? {}) },
          timeout: opts.timeoutMs ?? 30000,
          encoding: 'utf8',
        })
        done({ code: 0, stdout, stderr: '', truncated: false })
      } catch (e) {
        done({ code: e.status ?? 1, stdout: e.stdout ?? '', stderr: e.stderr ?? String(e), truncated: false })
      }
    }),
}
const extension = () => ({
  registerTool: (def) => {
    registered[def.name] = def
  },
  on: () => {},
  ctx: { log: { info: () => {} } },
})

kernelWorkflowTools.apply({ extension })

const expect = ['kernel_get_task', 'kernel_submit', 'kernel_get_job', 'kernel_get_task_status', 'kernel_read_feedback']
for (const name of expect) {
  if (!registered[name]) throw new Error(`tool not registered: ${name}`)
  const def = registered[name]
  for (const k of ['isReadOnly', 'isDestructive', 'isConcurrencySafe', 'isOpenWorld', 'replay', 'costHint', 'deferLoading', 'requiresApproval']) {
    if (!(k in def.meta)) throw new Error(`${name}: meta missing ${k}`)
  }
}
console.log('registered:', Object.keys(registered).join(', '))

// --- execute against the configured local run (read-only tools) ---
const call = (name, args) => registered[name].execute(args, fakeCtx)

const t = await call('kernel_get_task', { task_id: 'K12' })
if (t.isError) throw new Error(`kernel_get_task failed: ${t.content[0].text}`)
const digest = t.structured
if (!digest.budget || !digest.domain) throw new Error('digest missing budget/domain')
// The model's own read tool is fenced to the workspace; the contract text must
// ride inside kernel_get_task (2026-10-03 session: model got E_FS_DENIED trying
// to read tasks/K12/task.md itself).
if (typeof digest.contract !== 'string' || !digest.contract.includes('ModelNew')) {
  throw new Error('kernel_get_task must return the task.md contract text')
}
console.log('get_task ok: budget', JSON.stringify(digest.budget), '| contract', digest.contract.length, 'chars | workspace', digest.workspace)

const s = await call('kernel_get_task_status', { task_id: 'K12' })
if (s.isError) throw new Error(`kernel_get_task_status failed: ${s.content[0].text}`)
console.log('task_status ok: champion', s.structured.champion?.candidate_id ?? 'none',
  '| budget used', s.structured.budget?.source_candidates)

// Verify an escape path is rejected before kernel_submit reaches the ledger.
const esc = await call('kernel_submit', { task_id: 'K12', candidate_dir: '../escape' })
if (!esc.isError || !esc.content[0].text.includes('inside the session workspace')) {
  throw new Error('workspace boundary check did not fire')
}
console.log('submit boundary check ok')

// Regression pin for the 2026-10-03 E_FS_DENIED: every exec must start in the
// session workspace (the exec gate authorizes the cwd against the workspace
// fence; a cwd outside it - e.g. the kernel_lab root - is refused).
for (const call of execCalls) {
  if (call.cwd !== fakeCtx.cwd) {
    throw new Error(`exec ran with cwd ${call.cwd}, expected the session workspace ${fakeCtx.cwd}`)
  }
}
console.log(`exec cwd check ok (${execCalls.length} calls, all in session workspace)`)
console.log('all no-model checks passed')
