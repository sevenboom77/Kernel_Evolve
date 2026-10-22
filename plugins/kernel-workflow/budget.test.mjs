import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

async function fixture(config, used = 4) {
  const root = mkdtempSync(join(tmpdir(), 'kernel-tools-budget-'))
  const workspace = join(root, 'workspace')
  const cfg = join(root, 'config.json')
  writeFileSync(cfg, JSON.stringify({ kl_python: '/fake/python', kl_root: '/fake/kl', run_id: 'test', ...config }))
  process.env.KERNEL_TOOLS_CONFIG = cfg
  const mod = await import(`./index.mjs?fixture=${crypto.randomUUID()}`)
  const tools = {}
  mod.kernelWorkflowTools.apply({ extension: () => ({ registerTool(t) { tools[t.name] = t }, on() {}, ctx: { log: { info() {} } } }) })
  const commands = []
  const ctx = { cwd: workspace, timeoutMs: 30000, async exec(argv) {
    commands.push(argv[3])
    if (argv[3] === 'submit') {
      return { code: 0, stdout: JSON.stringify({ ok: true, job_id: 'job-test', candidate_id: 'candidate-test', source_hash: 'a'.repeat(64) }), stderr: '' }
    }
    return { code: 0, stdout: JSON.stringify({ ok: true, budget: { source_candidates: used, max_candidates: 12, remaining: 12 - used }, champion: null }), stderr: '' }
  } }
  return { mod, tools, ctx, commands, workspace, cfg, cleanup() { rmSync(root, { recursive: true, force: true }) } }
}

test('fixed search budget refuses the fifth candidate before submitting', async () => {
  const f = await fixture({ max_candidates: 4 })
  try {
    const result = await f.tools.kernel_submit.execute({ task_id: 'K12', candidate_dir: 'cand05' }, f.ctx)
    assert.equal(result.isError, true)
    assert.match(result.content[0].text, /BUDGET_EXHAUSTED/)
    assert.equal(f.commands.includes('submit'), false)
  } finally { f.cleanup() }
})

test('task status exposes the tighter experiment budget', async () => {
  const f = await fixture({ max_candidates: 4 }, 2)
  try {
    const result = await f.tools.kernel_get_task_status.execute({ task_id: 'K12' }, f.ctx)
    assert.equal(result.structured.budget.max_candidates, 4)
    assert.equal(result.structured.budget.remaining, 2)
  } finally { f.cleanup() }
})

test('expired experiment refuses a new submission', async () => {
  const f = await fixture({ max_candidates: 4, deadline_ms: Date.now() - 1000 }, 0)
  try {
    const result = await f.tools.kernel_submit.execute({ task_id: 'K12', candidate_dir: 'cand01' }, f.ctx)
    assert.equal(result.isError, true)
    assert.match(result.content[0].text, /DEADLINE/)
    assert.equal(f.commands.includes('submit'), false)
  } finally { f.cleanup() }
})

test('registered workspace cannot submit another task', async () => {
  const f = await fixture({ max_candidates: 4 }, 0)
  try {
    writeFileSync(f.cfg, JSON.stringify({ kl_python: '/fake/python', kl_root: '/fake/kl', run_id: 'test', workspaces: { [f.workspace]: 'K12' }, max_candidates: 4 }))
    const result = await f.tools.kernel_submit.execute({ task_id: 'K19', candidate_dir: 'cand01' }, f.ctx)
    assert.equal(result.isError, true)
    assert.match(result.content[0].text, /registered task/)
    assert.deepEqual(f.commands, [])
  } finally { f.cleanup() }
})

test('search preset bounds model steps and disables inner subagents', async () => {
  const f = await fixture({})
  try {
    assert.equal(f.mod.presets['kernel-search'].budget.max_steps, 24)
    assert.equal(f.mod.presets['kernel-search'].model.max_tokens, 4096)
    assert.equal(f.mod.presets['kernel-search'].subagent.max_depth, 0)
  } finally { f.cleanup() }
})

test('candidate in a workspace subdirectory reaches submission', async () => {
  const f = await fixture({ max_candidates: 4 }, 0)
  try {
    const result = await f.tools.kernel_submit.execute({ task_id: 'K12', candidate_dir: 'cand01' }, f.ctx)
    assert.notEqual(result.isError, true)
    assert.equal(result.structured.job_id, 'job-test')
    assert.deepEqual(f.commands, ['task-status', 'submit'])
  } finally { f.cleanup() }
})

test('parent traversal is refused before any backend command', async () => {
  const f = await fixture({ max_candidates: 4 }, 0)
  try {
    const result = await f.tools.kernel_submit.execute({ task_id: 'K12', candidate_dir: '../escape' }, f.ctx)
    assert.equal(result.isError, true)
    assert.match(result.content[0].text, /inside the session workspace/)
    assert.deepEqual(f.commands, [])
  } finally { f.cleanup() }
})

test('a sibling with the same workspace prefix is refused', async () => {
  const f = await fixture({ max_candidates: 4 }, 0)
  try {
    const result = await f.tools.kernel_submit.execute({ task_id: 'K12', candidate_dir: '../workspace-other/cand01' }, f.ctx)
    assert.equal(result.isError, true)
    assert.match(result.content[0].text, /inside the session workspace/)
    assert.deepEqual(f.commands, [])
  } finally { f.cleanup() }
})
