import type { ToolContext, ToolDef, ToolResult } from '@agnes/extension-api'
import { unavailableProjections } from '@agnes/extension-api'
import { Type } from '@sinclair/typebox'
import { describe, expect, it, vi } from 'vitest'
import { platformFacts } from '../src/effects/platform-facts.js'
import { artifactUri, toLedgerContent } from '../src/effects/tool-result.js'
import { HookEngine } from '../src/hooks/engine.js'
import { SessionHookPort } from '../src/hooks/port.js'
import { MemoryStorage } from '../src/log/memory-storage.js'
import { ToolRegistry } from '../src/registry/tools.js'
import { CompactionRunner } from '../src/step/compaction.js'
import { withTerminalJobFeedback } from '../src/step/job-feedback.js'
import { withPhase } from '../src/step/op-state.js'
import { presetDefaults } from '../src/step/preset.js'
import { fakeProvider, type Script, sent, textTurn, toolTurn, usage } from './helpers/fake-provider.js'
import { fakeSeams } from './helpers/fake-seams.js'
import { actor, openSession, readTool } from './helpers/open-session.js'

const ref = { sha256: 'a'.repeat(64), size: 3, mime: 'image/png' }

describe('durable DEV search memory in actual Core requests', () => {
  const hashes = { first: 'b'.repeat(64), repaired: 'c'.repeat(64) }
  const submit = (source = 'first') => toolTurn('kernel_submit', { task_id: 'K01', candidate_dir: source })
  const query = (source = 'first') => toolTurn('kernel_get_job', { job_id: `job-${source}`, wait_s: 0 })
  function payload(source = 'first', status = source === 'first' ? 'incorrect' : 'complete') {
    return {
      job: {
        job_id: `job-${source}`,
        candidate_id: `candidate-${source}`,
        task_id: 'K01',
        run_id: 'run-dev',
        source_hash: hashes[source as keyof typeof hashes],
        kind: 'candidate',
        phase: 'dev',
        status,
      },
      status,
      has_result: status !== 'running',
      result: {
        status,
        phase: 'dev',
        contaminated: false,
        ...(status === 'complete'
          ? { task_speedup: 2, below_measurement_floor: true, launch_bound: true }
          : {}),
        correctness: { ok: status === 'complete', passed: status === 'complete' ? 12 : 9, total: 12 },
      },
    }
  }
  const raw = (structured: unknown): ToolResult => ({
    content: [{ type: 'text', text: 'original terminal result' }],
    structured: structured as ToolResult['structured'],
  })
  const note = (
    request: { messages: Array<{ content: Array<{ type: string; text?: string }> }> } | undefined,
  ) =>
    request?.messages
      .flatMap((message) => message.content)
      .filter((block) => block.type === 'text' && block.text?.startsWith('[kernel search state]\n'))
      .at(-1)?.text ?? ''
  async function fixture(
    options: {
      scripts?: Script[]
      getJob?: ToolDef['execute']
      queryMeta?: Partial<ToolDef['meta']>
      seams?: ReturnType<typeof fakeSeams>
    } = {},
  ) {
    const registry = new ToolRegistry()
    const meta = (readTool() as ToolDef).meta
    registry.add(
      {
        name: 'kernel_submit',
        description: 'submit one source from this workspace',
        parameters: Type.Object({ task_id: Type.String(), candidate_dir: Type.String() }),
        meta: { ...meta, isReadOnly: false, replay: 'never', requiresApproval: 'never' },
        execute: async (args) => {
          const source = (args as { candidate_dir: string }).candidate_dir
          return raw({
            ok: true,
            job_id: `job-${source}`,
            candidate_id: `candidate-${source}`,
            source_hash: hashes[source as keyof typeof hashes],
            phase: 'dev',
            workspace: '/w',
          })
        },
      } as ToolDef,
      { source: 's', trust: 'builtin' },
    )
    registry.add(
      {
        name: 'kernel_get_job',
        description: 'query the actual submitted source',
        parameters: Type.Object({ job_id: Type.String(), wait_s: Type.Integer() }),
        meta: { ...meta, ...options.queryMeta },
        execute:
          options.getJob ?? (async (args) => raw(payload((args as { job_id: string }).job_id.slice(4)))),
      } as ToolDef,
      { source: 's', trust: 'builtin' },
    )
    const provider = fakeProvider(options.scripts ?? [submit(), query(), textTurn('done')])
    const opened = await openSession({
      provider,
      registry,
      ...(options.seams ? { seams: options.seams } : {}),
    })
    const commits: string[][] = []
    opened.session.onAppended((rows) => commits.push(rows.map((row) => row.type)))
    await opened.session.enqueue('next-turn', {
      content: [{ type: 'text', text: 'optimize current DEV task' }],
      actor,
    })
    return { ...opened, provider, registry, commits }
  }
  const run = (session: Awaited<ReturnType<typeof fixture>>['session']) =>
    session.run({ until: 'turn-end', signal: new AbortController().signal })
  async function advanceToMemory(f: Awaited<ReturnType<typeof fixture>>) {
    for (let step = 0; step < 24; step++) {
      if ((await f.log.scan({ type: 'x/core/kernel-search-observation', limit: 10 })).length) return
      const outcome = await f.session.step()
      if (outcome.phase === 'idle' || outcome.phase === 'terminal') break
    }
    expect(
      await f.log.scan({ type: 'x/core/kernel-search-observation', limit: 10 }),
      'actual tool phase did not persist terminal search memory',
    ).toHaveLength(1)
  }

  it('atomically records original results and updates the real provider state after a repaired source', async () => {
    const f = await fixture({
      scripts: [submit(), query(), submit('repaired'), query('repaired'), textTurn('done')],
    })
    expect((await run(f.session)).reason).toBe('completed')
    const observations = await f.log.scan({ type: 'x/core/kernel-search-observation', limit: 10 })
    expect(observations).toHaveLength(2)
    expect(observations[0]).toMatchObject({
      origin: 'system',
      trust: 'trusted',
      ignorable: true,
      data: { status: 'incorrect', passed: 9, total: 12, source_hash: hashes.first },
    })
    expect(f.commits.filter((batch) => batch.includes('x/core/kernel-search-observation'))).toEqual([
      expect.arrayContaining(['tool/result', 'effect/settled']),
      expect.arrayContaining(['tool/result', 'effect/settled']),
    ])
    const results = await f.log.scan({ type: 'tool/result', limit: 10 })
    expect(results[1]?.data).toMatchObject({ structured: payload(), isError: false })
    expect(results[3]?.data).toMatchObject({ structured: payload('repaired'), isError: false })
    const before = note(f.provider.requests[2])
    const after = note(f.provider.requests[4])
    expect(before).toContain(hashes.first)
    expect(before).toContain('repair_numerics_before_speed_changes')
    expect(after).toContain(hashes.repaired)
    expect(after).toContain('optimize_correct_DEV_source')
    expect(observations[1]?.data).toMatchObject({
      observed_dev_task_speedup: 2,
      below_measurement_floor: true,
      launch_bound: true,
    })
    expect(after).toContain('"observed_dev_task_speedup":2')
    expect(after).toContain('"best_observed_correct_DEV_source_index":1')
    expect(after).not.toBe(before)
    expect(f.provider.calls).toBe(5)
    expect(note(f.provider.requests[0])).toBe('')
  })

  it('deduplicates repeated queries without extra tool or provider calls', async () => {
    let queries = 0
    const f = await fixture({
      scripts: [submit(), query(), query(), textTurn('done')],
      getJob: async () => {
        queries++
        return raw(payload())
      },
    })
    await run(f.session)
    expect(queries).toBe(2)
    expect(f.provider.calls).toBe(4)
    expect(note(f.provider.requests.at(-1)).split(hashes.first).length - 1).toBe(1)
  })

  it('remembers the terminal outcome of the original bounded pending continuation', async () => {
    let queries = 0
    const f = await fixture({
      scripts: [submit(), toolTurn('kernel_get_job', { job_id: 'job-first', wait_s: 90 }), textTurn('done')],
      getJob: async () => raw(payload('first', ++queries === 1 ? 'running' : 'incorrect')),
    })
    await run(f.session)
    expect(queries).toBe(2)
    expect(await f.log.scan({ type: 'x/core/pending-job-observation', limit: 10 })).toHaveLength(1)
    expect(await f.log.scan({ type: 'x/core/kernel-search-observation', limit: 10 })).toHaveLength(1)
    expect(note(f.provider.requests.at(-1))).toContain('repair_numerics_before_speed_changes')
    expect(f.provider.calls).toBe(3)
  })

  it('keeps a failed tool verifier from creating trusted memory', async () => {
    const f = await fixture({
      seams: fakeSeams({
        verifier: {
          verify: async (scope) => ({ verdict: scope === 'tool' ? 'fail' : 'pass', reasons: [] }),
        },
      }),
    })
    await run(f.session)
    expect(await f.log.scan({ type: 'x/core/kernel-search-submission', limit: 10 })).toHaveLength(0)
    expect(await f.log.scan({ type: 'x/core/kernel-search-observation', limit: 10 })).toHaveLength(0)
  })

  it('clears the previous turn state rather than inheriting a candidate', async () => {
    const f = await fixture()
    await run(f.session)
    expect(note(f.provider.requests.at(-1))).toContain(hashes.first)
    await f.session.enqueue('next-turn', { content: [{ type: 'text', text: 'a fresh search' }], actor })
    await run(f.session)
    expect(note(f.provider.requests.at(-1))).toBe('[kernel search state]\n(none)')
  })

  it.each(['kernel_submit', 'kernel_get_job'])('keeps a %s result override authoritative', async (tool) => {
    const f = await fixture()
    f.session.hooks = {
      ...f.session.hooks,
      toolResult: async (input) =>
        input.name === tool
          ? { result: { ...input.result, content: [{ type: 'text', text: 'hook result' }] } }
          : {},
    }
    await run(f.session)
    expect(await f.log.scan({ type: 'x/core/kernel-search-observation', limit: 10 })).toHaveLength(0)
    expect(note(f.provider.requests.at(-1))).toBe('')
  })

  it('persists memory through the actual no-op SessionHookPort used by the native runtime', async () => {
    const f = await fixture()
    const platform = platformFacts(fakeSeams().platform)
    const engine = new HookEngine({ diag: () => undefined, onFailure: () => undefined, platform })
    const port = new SessionHookPort(engine, {
      context: () => ({
        session: { key: f.session.key, lane: f.session.lane, workspaceRoot: '/w' },
        projections: unavailableProjections,
        replayed: false,
        signal: new AbortController().signal,
        lease: { expiresAt: '2099-01-01T00:00:00Z', scope: {}, budget: { remaining: 10 } },
        log: f.session.d.logger,
        platform,
      }),
      budget: () => ({ remaining: 24, cap: 24 }),
      surface: () => [],
      surfaceDigest: () => ({ nodes: 0, tokensEstimate: 0 }),
      verifierTier: () => 0,
      contextOverflow: () => undefined,
      compactPlanIgnored: () => undefined,
    })
    f.session.hooks = { ...f.session.hooks, toolResult: port.toolResult.bind(port) }
    await run(f.session)
    expect(await f.log.scan({ type: 'x/core/kernel-search-observation', limit: 10 })).toHaveLength(1)
    expect(note(f.provider.requests.at(-1))).toContain('repair_numerics_before_speed_changes')
    expect(f.provider.calls).toBe(3)
  })

  it('excludes a raw result changed in place by a custom Core hook', async () => {
    const f = await fixture()
    f.session.hooks = {
      ...f.session.hooks,
      toolResult: async (input) => {
        if (input.name === 'kernel_get_job')
          input.result.content.push({ type: 'text', text: 'mutated result' })
        return { result: input.result }
      },
    }
    await run(f.session)
    expect(await f.log.scan({ type: 'x/core/kernel-search-observation', limit: 10 })).toHaveLength(0)
  })

  it.each([
    'unknown job',
    'wrong task',
    'held phase',
    'contaminated',
    'transport failure',
    'untrusted policy',
  ])('does not remember %s as current DEV search state', async (bad) => {
    const f = await fixture({
      queryMeta: bad === 'untrusted policy' ? { isOpenWorld: true } : {},
      getJob: async () => {
        if (bad === 'transport failure') throw new Error('fixture transport unavailable')
        const data = payload()
        if (bad === 'unknown job') data.job.job_id = 'job-other'
        if (bad === 'wrong task') data.job.task_id = 'K02'
        if (bad === 'held phase') data.result.phase = 'holdout'
        if (bad === 'contaminated') data.result.contaminated = true
        return raw(data)
      },
    })
    await run(f.session)
    expect(await f.log.scan({ type: 'x/core/kernel-search-observation', limit: 10 })).toHaveLength(0)
    expect(note(f.provider.requests.at(-1))).toBe('')
  })

  it.each(['incorrect', 'complete'])(
    'reconstructs %s state after reopening an actual completed tool commit',
    async (status) => {
      const f = await fixture({ getJob: async () => raw(payload('first', status)) })
      await advanceToMemory(f)
      const rows = await f.log.scan({ fromSeq: 1, toSeq: f.log.lastSeq, limit: 10_000 })
      const storage = MemoryStorage.fromEvents('k', rows, { opCells: f.opCellsBefore(f.log.lastSeq + 1) })
      const provider = fakeProvider([textTurn('recovered')])
      const reopened = await openSession({ storage, registry: f.registry, provider, writerRunId: 'reopened' })
      await reopened.session.resume()
      expect((await run(reopened.session)).reason).toBe('completed')
      expect(provider.calls).toBe(1)
      expect(note(provider.requests[0])).toContain(hashes.first)
      expect(note(provider.requests[0])).toContain(
        status === 'complete' ? '"observed_dev_task_speedup":2' : 'repair_numerics_before_speed_changes',
      )
      expect(await reopened.log.scan({ type: 'tool/result', limit: 10 })).toHaveLength(2)
    },
  )

  it('keeps both sources when concurrent terminal results commit in one model batch', async () => {
    const both: Script = [
      sent(),
      {
        type: 'toolcall_end',
        call: { toolUseId: '', name: 'kernel_get_job', args: { job_id: 'job-first', wait_s: 0 }, ordinal: 0 },
        via: 'native',
      },
      {
        type: 'toolcall_end',
        call: {
          toolUseId: '',
          name: 'kernel_get_job',
          args: { job_id: 'job-repaired', wait_s: 0 },
          ordinal: 1,
        },
        via: 'native',
      },
      usage(),
      { type: 'done', reason: 'toolUse' },
    ]
    const f = await fixture({ scripts: [submit(), submit('repaired'), both, textTurn('done')] })
    await run(f.session)
    expect(await f.log.scan({ type: 'x/core/kernel-search-observation', limit: 10 })).toHaveLength(2)
    expect(note(f.provider.requests[3])).toContain(hashes.first)
    expect(note(f.provider.requests[3])).toContain(hashes.repaired)
    expect(f.provider.calls).toBe(4)
  })

  it.each(['incorrect', 'complete'])(
    'rebuilds %s from the ledger after actual context compaction in the current turn',
    async (status) => {
      const f = await fixture({
        getJob: async () => raw(payload('first', status)),
        scripts: [submit(), query(), textTurn('summary without source details'), textTurn('done')],
      })
      await advanceToMemory(f)
      f.session.compaction = new CompactionRunner({
        plan: async (input) => {
          const nodes = input.getSurface()
          const first = nodes[0]
          const end = nodes.at(-3)
          const last = nodes.at(-2)
          if (!first || !end || !last) throw new Error('current-turn compactable surface required')
          return {
            keepFromSeq: last.seq,
            summarizeRange: [first.seq, end.seq],
            prompts: { system: 'summarize current history', history: 'omit source details' },
            maxTokens: 100,
            details: { readFiles: [], modifiedFiles: [] },
          }
        },
        onCompact: async () => undefined,
      })
      const op = f.session.op()
      if (!op) throw new Error('search must remain open')
      await f.session.transition(
        [],
        withPhase(op, { kind: 'compaction', reason: 'requested', resumeAfter: op.phase }),
      )
      expect((await run(f.session)).reason).toBe('completed')
      expect(
        await f.log.scan({ type: 'x/core/compaction-end', limit: 10 }),
        JSON.stringify(await f.log.scan({ type: 'x/core/compaction-failed', limit: 10 })),
      ).toHaveLength(1)
      expect(note(f.provider.requests.at(-1))).toContain(hashes.first)
      expect(note(f.provider.requests.at(-1))).toContain(
        status === 'complete' ? '"observed_dev_task_speedup":2' : 'repair_numerics_before_speed_changes',
      )
      expect(f.provider.requests.filter((request) => request.kind === 'inference')).toHaveLength(3)
    },
  )
})

describe('bounded source-bound pending job queries', () => {
  const args = { job_id: 'job-pending', wait_s: 90 }
  const payload = (status = 'running', hasResult = false) => ({
    job: { job_id: args.job_id, source_hash: 'c'.repeat(64), kind: 'candidate', phase: 'dev', status },
    status,
    has_result: hasResult,
    result: hasResult ? { status } : {},
  })
  const result = (structured = payload()): ToolResult => ({
    content: [{ type: 'text', text: JSON.stringify(structured) }],
    structured,
  })
  const tool = (
    execute: ToolDef['execute'],
    meta: Partial<ToolDef['meta']> = {},
    name = 'kernel_get_job',
  ) => ({
    name,
    description: 'query the same job with an explicit bounded wait',
    parameters: Type.Object({ job_id: Type.String(), wait_s: Type.Integer() }),
    meta: { ...(readTool() as ToolDef).meta, ...meta },
    execute,
  })
  async function fixture(
    execute: ToolDef['execute'],
    options: {
      meta?: Partial<ToolDef['meta']>
      name?: string
      args?: typeof args
      timeoutMs?: number
    } = {},
  ) {
    const registry = new ToolRegistry()
    registry.add(tool(execute, options.meta, options.name) as ToolDef, { source: 's', trust: 'builtin' })
    const provider = fakeProvider([
      toolTurn(options.name ?? 'kernel_get_job', options.args ?? args),
      textTurn('done'),
    ])
    const preset = presetDefaults()
    preset.tools.timeoutMs = options.timeoutMs ?? 120_000
    const opened = await openSession({ registry, provider, preset })
    await opened.session.enqueue('next-turn', { content: [{ type: 'text', text: 'query' }], actor })
    return { ...opened, provider }
  }
  const run = (
    session: Awaited<ReturnType<typeof fixture>>['session'],
    signal = new AbortController().signal,
  ) => session.run({ until: 'turn-end', signal })

  it('retains the raw pending observation and returns the second terminal result to the next provider', async () => {
    const first = result()
    const final = result(payload('complete', true))
    const contexts: ToolContext[] = []
    const f = await fixture(async (_args, context) => {
      contexts.push(context)
      return contexts.length === 1 ? first : final
    })
    await run(f.session)
    const rows = await f.log.scan({ type: 'tool/result', limit: 10 })
    expect(rows).toHaveLength(1)
    expect(rows[0]?.data).toMatchObject({ content: final.content, structured: final.structured })
    const observations = await f.log.scan({ type: 'x/core/pending-job-observation', limit: 10 })
    expect(observations).toHaveLength(1)
    expect(observations[0]?.data).toMatchObject({ content: first.content, structured: first.structured })
    expect(contexts).toHaveLength(2)
    const [firstContext, secondContext] = contexts
    if (!firstContext || !secondContext) throw new Error('both bounded query contexts must exist')
    expect(secondContext.timeoutMs).toBeLessThanOrEqual(firstContext.timeoutMs)
    expect(contexts[1]?.signal).toBe(contexts[0]?.signal)
    expect(f.provider.calls).toBe(2)
    expect(JSON.stringify(f.provider.requests[1]?.messages)).toContain('complete')
  })

  it('returns the second pending result without a third continuation', async () => {
    let queries = 0
    const f = await fixture(async () => {
      queries++
      return result()
    })
    await run(f.session)
    expect(queries).toBe(2)
    expect((await f.log.scan({ type: 'tool/result', limit: 10 }))[0]?.data).toMatchObject({
      structured: payload(),
    })
  })

  it.each([
    ['nonwaiting', { wait_s: 0 }],
    ['out-of-range wait', { wait_s: 91 }],
    ['another tool', { name: 'query_other_job' }],
    ['mutation policy', { meta: { isReadOnly: false } }],
    ['destructive policy', { meta: { isDestructive: true } }],
    ['no replay', { meta: { replay: 'never' } }],
    ['short original timeout', { timeoutMs: 29_000 }],
  ] as const)('does not continue %s', async (_label, options) => {
    let queries = 0
    const f = await fixture(
      async () => {
        queries++
        return result()
      },
      {
        ...options,
        args: { ...args, ...('wait_s' in options ? { wait_s: options.wait_s } : {}) },
      },
    )
    await run(f.session)
    expect(queries).toBe(1)
    expect(await f.log.scan({ type: 'x/core/pending-job-observation', limit: 10 })).toHaveLength(0)
  })

  it.each([
    ['wrong job', { job: { ...payload().job, job_id: 'job-other' } }],
    ['wrong source', { job: { ...payload().job, source_hash: 'invalid' } }],
    ['baseline', { job: { ...payload().job, kind: 'baseline' } }],
    ['held phase', { job: { ...payload().job, phase: 'holdout' } }],
    ['discordant status', { job: { ...payload().job, status: 'queued' } }],
    ['queued state', { status: 'queued', job: { ...payload().job, status: 'queued' } }],
    ['already has result', { has_result: true }],
    ['terminal', { status: 'complete', job: { ...payload().job, status: 'complete' } }],
  ])('preserves %s observations without continuing', async (_label, change) => {
    let queries = 0
    const first = result({ ...payload(), ...change })
    const f = await fixture(async () => {
      queries++
      return first
    })
    await run(f.session)
    expect(queries).toBe(1)
    expect((await f.log.scan({ type: 'tool/result', limit: 10 }))[0]?.data).toMatchObject({
      content: first.content,
      structured: first.structured,
    })
    expect(await f.log.scan({ type: 'x/core/pending-job-observation', limit: 10 })).toHaveLength(0)
    if (_label === 'queued state') {
      expect(JSON.stringify(f.provider.requests[1]?.messages)).toContain('queued')
    }
  })

  it('preserves an error result without continuing', async () => {
    let queries = 0
    const first = { ...result(), isError: true }
    const f = await fixture(async () => {
      queries++
      return first
    })
    await run(f.session)
    expect(queries).toBe(1)
    expect((await f.log.scan({ type: 'tool/result', limit: 10 }))[0]?.data).toMatchObject({
      isError: true,
      structured: first.structured,
    })
  })

  it.each([
    ['termination', { ...result(), terminate: true }],
    ['present deferred marker', { ...result(), deferred: undefined }],
    ['malformed ordinary result', null],
  ])('does not continue a %s result', async (_label, first) => {
    let queries = 0
    const f = await fixture(async () => {
      queries++
      return first as ToolResult
    })
    await run(f.session)
    expect(queries).toBe(1)
    expect(await f.log.scan({ type: 'x/core/pending-job-observation', limit: 10 })).toHaveLength(0)
  })

  it('honors cancellation during the continuation while retaining the first observation', async () => {
    let queries = 0
    const ac = new AbortController()
    const f = await fixture(async (_args, context) => {
      if (++queries === 2) {
        ac.abort()
        expect(context.signal.aborted).toBe(true)
        throw new Error('query cancelled')
      }
      return result()
    })
    await run(f.session, ac.signal)
    expect(queries).toBe(2)
    expect((await f.log.scan({ type: 'x/core/pending-job-observation', limit: 10 }))[0]?.data).toMatchObject({
      structured: payload(),
    })
  })

  it('does not continue when the first pending observation cannot be durably recorded', async () => {
    let queries = 0
    const f = await fixture(async () => {
      queries++
      return result()
    })
    const append = f.log.append.bind(f.log)
    vi.spyOn(f.log, 'append').mockImplementation(async (...params) => {
      if (params[0].some((event) => event.type === 'x/core/pending-job-observation'))
        throw new Error('diagnostic storage unavailable')
      return append(...params)
    })
    await run(f.session)
    expect(queries).toBe(1)
    expect((await f.log.scan({ type: 'tool/result', limit: 10 }))[0]?.data).toMatchObject({
      structured: payload(),
    })
  })

  it('honors cancellation before the second query', async () => {
    let queries = 0
    const ac = new AbortController()
    const f = await fixture(async () => {
      queries++
      ac.abort()
      return result()
    })
    await run(f.session, ac.signal)
    expect(queries).toBe(1)
  })

  it('retains the first observation when the continuation fails', async () => {
    let queries = 0
    const f = await fixture(async () => {
      if (++queries === 2) throw new Error('second readonly query failed')
      return result()
    })
    await run(f.session)
    expect(queries).toBe(2)
    expect((await f.log.scan({ type: 'tool/result', limit: 10 }))[0]?.data).toMatchObject({ isError: true })
    expect((await f.log.scan({ type: 'x/core/pending-job-observation', limit: 10 }))[0]?.data).toMatchObject({
      structured: payload(),
    })
  })

  it('keeps the existing extension result override after the actual terminal query', async () => {
    let queries = 0
    const f = await fixture(async () => result(++queries === 1 ? payload() : payload('complete', true)))
    let seen: unknown
    f.session.hooks = {
      ...f.session.hooks,
      toolResult: async (input) => {
        seen = input.result.structured
        return { result: { content: [{ type: 'text', text: 'extension final result' }] } }
      },
    }
    await run(f.session)
    expect(seen).toEqual(payload('complete', true))
    expect((await f.log.scan({ type: 'tool/result', limit: 10 }))[0]?.data).toMatchObject({
      content: [{ type: 'text', text: 'extension final result' }],
    })
  })
})

describe('toLedgerContent', () => {
  it('maps author blocks to protocol ContentBlock', () => {
    expect(
      toLedgerContent([
        { type: 'text', text: 'hi' },
        { type: 'image', ref, mime: 'image/png' },
        { type: 'ref', ref: { ...ref, mime: 'text/csv' } },
      ]),
    ).toEqual([
      { type: 'text', text: 'hi' },
      { type: 'resource_link', uri: `artifact://${'a'.repeat(64)}`, mimeType: 'image/png', name: 'image' },
      { type: 'resource_link', uri: `artifact://${'a'.repeat(64)}`, mimeType: 'text/csv', name: 'artifact' },
    ])
    expect(artifactUri(ref)).toBe(`artifact://${'a'.repeat(64)}`)
  })

  // ArtifactRef carries `mime` as a required key, so a `ref` block with no block-level mime still
  // gets one from the ref. Walked here because it is the shape base's tools actually produce: an
  // artifact handed back without restating its type.
  it('a ref block with no block-level mime takes the mime off the ref', () => {
    expect(toLedgerContent([{ type: 'ref', ref }])).toEqual([
      { type: 'resource_link', uri: `artifact://${'a'.repeat(64)}`, mimeType: 'image/png', name: 'artifact' },
    ])
  })

  // The block-level mime is what the author said this use of the artifact is, so it wins over the
  // type the artifact was stored under. Walked because the two disagree only here, and a converter
  // that read the ref first would look right in every other case.
  it('a block-level mime overrides the one on the ref', () => {
    expect(toLedgerContent([{ type: 'ref', ref, mime: 'text/csv' }])).toEqual([
      { type: 'resource_link', uri: `artifact://${'a'.repeat(64)}`, mimeType: 'text/csv', name: 'artifact' },
    ])
  })
})

describe('the tools phase records converted blocks', () => {
  // The conversion is only worth anything where it is wired in. A tool handing back an artifact is
  // the case the author shape exists for, and the row it lands on is what the next request shows the
  // model - so the assertion is on the ledger, not on the converter called a second time.
  it('an artifact a tool returns reaches the ledger as a resource_link', async () => {
    const registry = new ToolRegistry()
    registry.add(
      {
        name: 'read',
        description: 'read',
        parameters: Type.Object({}),
        meta: {
          isReadOnly: true,
          isDestructive: false,
          isConcurrencySafe: true,
          isOpenWorld: false,
          replay: 'safe',
          costHint: undefined,
          deferLoading: undefined,
          requiresApproval: undefined,
        },
        execute: async () => ({ content: [{ type: 'ref', ref, mime: 'text/csv' }] }),
      } as never,
      { source: 's', trust: 'builtin' },
    )
    const { session, log } = await openSession({
      provider: fakeProvider([toolTurn('read', { x: 1 }), textTurn('done')]),
      registry,
    })
    await session.enqueue('next-turn', { content: [{ type: 'text', text: 'go' }], actor })
    await session.run({ until: 'turn-end', signal: new AbortController().signal })
    const rows = await log.scan({ type: 'tool/result', limit: 5 })
    expect((rows[0]?.data as { content?: unknown })?.content).toEqual([
      { type: 'resource_link', uri: `artifact://${'a'.repeat(64)}`, mimeType: 'text/csv', name: 'artifact' },
    ])
  })

  it('records only the public structured result and never UI-only details', async () => {
    const registry = new ToolRegistry()
    const def = readTool() as ToolDef
    def.execute = async () => ({
      content: [{ type: 'text', text: 'human result' }],
      structured: { rows: 2, source: 'public' },
      details: { selectedTab: 'private-ui' },
    })
    registry.add(def, { source: 's', trust: 'builtin' })
    const { session, log } = await openSession({
      provider: fakeProvider([toolTurn('read', {}), textTurn('done')]),
      registry,
    })
    await session.enqueue('next-turn', { content: [{ type: 'text', text: 'go' }], actor })
    await session.run({ until: 'turn-end', signal: new AbortController().signal })
    const data = (await log.scan({ type: 'tool/result', limit: 5 }))[0]?.data
    expect(data).toMatchObject({ structured: { rows: 2, source: 'public' } })
    expect(JSON.stringify(data)).not.toContain('private-ui')
  })
})

describe('the toolResult hook can override what the tools phase records and returns', () => {
  it('passes the real enforcement shape and applies the override to the ledger row', async () => {
    const registry = new ToolRegistry()
    registry.add(readTool(), { source: 's', trust: 'builtin' })
    const { session, log } = await openSession({
      provider: fakeProvider([toolTurn('read', { path: 'a' }), textTurn('done')]),
      registry,
    })
    let seenEnforcement: unknown
    let seenArgs: unknown
    session.hooks = {
      ...session.hooks,
      toolResult: async (p) => {
        seenEnforcement = p.enforcement
        seenArgs = p.args
        return { result: { content: [{ type: 'text', text: 'overridden' }] } }
      },
    }
    await session.enqueue('next-turn', { content: [{ type: 'text', text: 'go' }], actor })
    await session.run({ until: 'turn-end', signal: new AbortController().signal })
    // The real shape `s.d.runtime.enforcement()` returns — {level, scope} — not the placeholder the
    // design's sample code left in `{ level: enforcement(), scope: [] }`.
    expect(seenEnforcement).toEqual({ level: 'full', scope: ['file', 'network', 'process'] })
    expect(seenArgs).toEqual({ path: 'a' })
    const rows = await log.scan({ type: 'tool/result', limit: 5 })
    expect((rows[0]?.data as { content?: unknown })?.content).toEqual([{ type: 'text', text: 'overridden' }])
  })
})

describe('terminal source-bound job recovery feedback', () => {
  const failure = (status = 'compile_fail') => ({
    job: { job_id: 'job-1', source_hash: 'b'.repeat(64), status },
    status,
    has_result: true,
    result: { status, error: 'original compiler diagnostic' },
  })
  const numericalFailure = (correctness: Record<string, unknown> = {}) => ({
    ...failure('incorrect'),
    result: {
      status: 'incorrect',
      error: 'original numerical diagnostic',
      correctness: {
        ok: false,
        passed: 11,
        total: 12,
        first_failures: [{ case_id: 'case-8', problems: ['output mismatch at original index'] }],
        ...correctness,
      },
    },
  })

  async function run(result: ToolResult, override = false) {
    const registry = new ToolRegistry()
    const def = readTool() as ToolDef
    def.execute = async () => result
    registry.add(def, { source: 's', trust: 'builtin' })
    const provider = fakeProvider([toolTurn('read', {}), textTurn('done')])
    const { session, log } = await openSession({ provider, registry })
    if (override)
      session.hooks = {
        ...session.hooks,
        toolResult: async () => ({ result: { content: [{ type: 'text', text: 'extension override' }] } }),
      }
    await session.enqueue('next-turn', { content: [{ type: 'text', text: 'go' }], actor })
    await session.run({ until: 'turn-end', signal: new AbortController().signal })
    return {
      row: (await log.scan({ type: 'tool/result', limit: 5 }))[0]?.data,
      request: provider.requests.at(-1),
    }
  }

  it.each(['compile_fail', 'incorrect', 'timeout', 'oom', 'crashed'])(
    'explains a completed %s job while preserving the successful query and diagnostic',
    async (status) => {
      const structured = failure(status)
      const original = { type: 'text' as const, text: JSON.stringify(structured) }
      const observed = await run({ content: [original], structured, isError: false })
      expect(observed.row).toMatchObject({
        structured,
        isError: false,
        content: [
          original,
          { type: 'text', text: expect.stringContaining('The queried job has finished with a failure') },
        ],
      })
      expect(JSON.stringify(observed.request)).toContain('inspect its diagnostics')
      expect(JSON.stringify(observed.request)).toContain('original compiler diagnostic')
    },
  )

  it.each([
    "AttributeError: module 'example.math' has no attribute 'missing'",
    "NameError: name 'missing_kernel' is not defined",
    'CompilationError: at 3:2: unsupported expression',
    'Traceback (most recent call last):\n  original compiler diagnostic',
  ])('points to the embedded diagnostic without requiring another artifact read: %s', async (error) => {
    const structured = { ...failure(), result: { status: 'compile_fail', error } }
    const original = { type: 'text' as const, text: JSON.stringify(structured) }
    const observed = await run({ content: [original], structured, isError: false })
    expect(observed.row).toMatchObject({
      structured,
      isError: false,
      content: [
        original,
        {
          type: 'text',
          text: expect.stringContaining('diagnostic is already included'),
        },
      ],
    })
    expect(JSON.stringify(observed.request)).toContain('only if the included error is insufficient')
    expect(JSON.stringify(observed.request)).toContain(error.split('\n')[0])
  })

  it.each([
    'CompilationError: at 22:15:\n\n    acc += tl.dot(b, a[:, None])\n               ^',
    'CompilationError: at 3:2:\r\n    # TypeError: this is only a source comment\r\n    invalid()\r\n    ~^~\r\n',
  ])('distinguishes a compiler location excerpt from an underlying cause', async (error) => {
    const structured = { ...failure(), result: { status: 'compile_fail', error } }
    const original = { type: 'text' as const, text: JSON.stringify(structured) }
    const observed = await run({ content: [original], structured, isError: false })
    expect(observed.row).toMatchObject({
      structured,
      isError: false,
      content: [original, expect.any(Object)],
    })
    const content = (observed.row as { content: { text?: string }[] }).content
    expect(content[1]?.text).toContain('omits the underlying cause')
    expect(content[1]?.text).toContain('full error or traceback diagnostics before revising')
    expect(content[1]?.text).not.toContain('diagnostic is already included')
    expect(content[1]?.text).not.toContain('tl.dot')
    expect(content[1]?.text).not.toContain('TypeError')
    expect(JSON.stringify(observed.request)).toContain('omits the underlying cause')
  })

  it.each([
    'CompilationError: at 3:2:\n    invalid()\n    ^\nTypeError: unsupported expression',
    "CompilationError: at 15:21:\n    x = unknown_name\n        ^\nNameError('unknown_name is not defined')",
    'CompilationError: at 3:2: unsupported expression',
  ])('keeps an included compiler cause available without an obligatory extra read', async (error) => {
    const structured = { ...failure(), result: { status: 'compile_fail', error } }
    const observed = await run({ content: [{ type: 'text', text: error }], structured, isError: false })
    const content = (observed.row as { content: { text?: string }[] }).content
    expect(content[1]?.text).toContain('diagnostic is already included')
    expect(content[1]?.text).not.toContain('omits the underlying cause')
  })

  it.each([
    'Traceback (most recent call last):\n  File "compiler.py", line 1, in load\nValueError: Unsupported ptr type <[64], int32> in tl.load\n\nThe above exception was the direct cause of the following exception:\n\nTraceback (most recent call last):\ntriton.compiler.errors.CompilationError: at 3:2:\n    load()\n    ^',
    'Traceback (most recent call last):\r\n  File "compiler.py", line 1, in build\r\ncompiler.types.TypeError: unsupported expression\r\n\r\nCompilationError: at 3:2:\r\n    build()\r\n    ^',
  ])('uses an underlying compiler cause already included in the traceback', async (traceback) => {
    const structured = {
      ...failure(),
      result: { status: 'compile_fail', error: 'CompilationError: at 3:2:\n    load()\n    ^', traceback },
    }
    const original = { type: 'text' as const, text: JSON.stringify(structured) }
    const observed = await run({ content: [original], structured, isError: false })
    expect(observed.row).toMatchObject({
      structured,
      isError: false,
      content: [original, expect.any(Object)],
    })
    const content = (observed.row as { content: { text?: string }[] }).content
    expect(content[1]?.text).toContain('underlying compiler cause is already included in the traceback')
    expect(content[1]?.text).toContain('only if the included traceback is insufficient')
    expect(content[1]?.text).not.toContain('omits the underlying cause')
    expect(content[1]?.text).not.toContain('Unsupported ptr')
    expect(content[1]?.text).not.toContain('unsupported expression')
    expect(JSON.stringify(observed.request)).toContain(
      'underlying compiler cause is already included in the traceback',
    )
    expect(JSON.stringify(observed.request)).toContain('CompilationError')
  })

  it.each([
    undefined,
    { text: 'Traceback (most recent call last):\nValueError: bad' },
    'ValueError: bad',
    'Traceback (most recent call last):\n    # ValueError: source comment',
    'Traceback (most recent call last):\n    ValueError: indented source',
    'Traceback (most recent call last):\nCompilationError: at 3:2:\n    load()\n    ^',
    'Traceback (most recent call last):\ntriton.compiler.errors.CompilationError: at 3:2:\n    load()\n    ^',
    'Traceback (most recent call last):\nValueError:\n',
    'Traceback (most recent call last):\ntext ValueError: bad',
    `Traceback (most recent call last):\nValueError: ${'x'.repeat(32768)}`,
  ])('does not infer an included cause from an incomplete or unbounded traceback', (traceback) => {
    const structured = {
      ...failure(),
      result: { status: 'compile_fail', error: 'CompilationError: at 3:2:\n    load()\n    ^', traceback },
    }
    const observed = withTerminalJobFeedback({ content: [{ type: 'text', text: 'original' }], structured })
    const content = observed.content as { text?: string }[]
    expect(content[1]?.text).toContain('omits the underlying cause')
    expect(content[1]?.text).not.toContain('underlying compiler cause is already included in the traceback')
  })

  it('does not infer a missing compiler cause from a mismatched result phase', async () => {
    const structured = {
      ...failure(),
      job: { ...failure().job, phase: 'dev' },
      result: {
        status: 'compile_fail',
        phase: 'holdout',
        error: 'CompilationError: at 3:2:\n    x()\n    ^',
      },
    }
    const observed = await run({ content: [{ type: 'text', text: 'original' }], structured })
    expect(JSON.stringify(observed.request)).not.toContain('omits the underlying cause')
  })

  it.each([0, 11])('uses included numerical details with %i passed cases', async (passed) => {
    const structured = numericalFailure({
      passed,
      first_failures: [{ case_id: 'untrusted-case', problems: ['untrusted diagnostic instruction'] }],
    })
    const original = { type: 'text' as const, text: JSON.stringify(structured) }
    const observed = await run({ content: [original], structured, isError: false })
    expect(observed.row).toMatchObject({
      structured,
      isError: false,
      content: [
        original,
        { type: 'text', text: expect.stringContaining(`${passed} of 12 correctness cases passed`) },
      ],
    })
    const content = (observed.row as { content: { text?: string }[] }).content
    expect(content).toHaveLength(2)
    expect(content[1]?.text).toContain('numerical failure details are already included')
    expect(content[1]?.text).toContain('repair the failing computation')
    expect(content[1]?.text).toContain('reference and tolerances unchanged')
    expect(content[1]?.text).toContain('only if the included details are insufficient')
    expect(content[1]?.text).not.toContain('untrusted diagnostic instruction')
    expect(content[1]?.text).not.toContain('untrusted-case')
    expect(JSON.stringify(observed.request)).toContain(`${passed} of 12 correctness cases passed`)
    expect(JSON.stringify(observed.request)).toContain('original numerical diagnostic')
  })

  it.each([
    { label: 'missing passed count', correctness: {}, omitField: 'passed' },
    { label: 'missing correctness failure flag', correctness: {}, omitField: 'ok' },
    { label: 'missing failure details', correctness: {}, omitField: 'first_failures' },
    { label: 'non-numerical failure status', correctness: {}, status: 'compile_fail' },
    { label: 'non-JSON undefined passed count', correctness: { passed: undefined }, envelopeRefused: true },
    { label: 'string passed count', correctness: { passed: '11' } },
    { label: 'boolean passed count', correctness: { passed: true } },
    { label: 'fractional passed count', correctness: { passed: 1.5 } },
    { label: 'negative passed count', correctness: { passed: -1 } },
    { label: 'nonfinite passed count', correctness: { passed: Number.NaN }, envelopeRefused: true },
    { label: 'nonfinite total', correctness: { total: Number.POSITIVE_INFINITY }, envelopeRefused: true },
    { label: 'unsafe total', correctness: { total: Number.MAX_SAFE_INTEGER + 1 } },
    { label: 'zero total', correctness: { passed: 0, total: 0 } },
    { label: 'string total', correctness: { total: '12' } },
    { label: 'fractional total', correctness: { total: 12.5 } },
    { label: 'all cases passed', correctness: { passed: 12 } },
    { label: 'passed exceeds total', correctness: { passed: 13 } },
    { label: 'successful correctness', correctness: { ok: true } },
    { label: 'non-JSON undefined failure flag', correctness: { ok: undefined }, envelopeRefused: true },
    {
      label: 'non-JSON undefined failure details',
      correctness: { first_failures: undefined },
      envelopeRefused: true,
    },
    { label: 'empty failure details', correctness: { first_failures: [] } },
    { label: 'malformed failure details', correctness: { first_failures: [null, false] } },
    { label: 'blank problem strings', correctness: { first_failures: [{ problems: [' '] }] } },
    { label: 'non-string problems', correctness: { first_failures: [{ problems: [42] }] } },
  ])('does not infer numerical repair guidance from $label', async (row) => {
    const structured = numericalFailure(row.correctness)
    if ('omitField' in row) Reflect.deleteProperty(structured.result.correctness, row.omitField)
    if ('status' in row) {
      structured.status = row.status
      structured.job.status = row.status
      structured.result.status = row.status
    }
    const original = { type: 'text' as const, text: JSON.stringify(structured) }
    const observed = await run({ content: [original], structured })
    if ('envelopeRefused' in row) {
      expect(observed.row).toMatchObject({
        isError: true,
        content: [{ type: 'text', text: expect.stringContaining('E_ENVELOPE') }],
      })
      expect(JSON.stringify(observed.request)).not.toContain('correctness cases passed')
      return
    }
    expect(observed.row).toMatchObject({ structured, content: [original, expect.any(Object)] })
    const content = (observed.row as { content: { text?: string }[] }).content
    expect(content[1]?.text).toContain('inspect its diagnostics')
    expect(content[1]?.text).not.toContain('correctness cases passed')
  })

  it.each([
    {
      label: 'successful result with historical failure',
      data: { ...failure('complete'), history: [failure()] },
    },
    { label: 'running result', data: { ...failure('running'), has_result: false } },
    { label: 'unknown result', data: failure('inconclusive') },
    { label: 'missing identity', data: { ...failure(), job: { status: 'compile_fail' } } },
    {
      label: 'invalid source hash',
      data: { ...failure(), job: { ...failure().job, source_hash: 'not-a-hash' } },
    },
    { label: 'different job status', data: { ...failure(), job: { ...failure().job, status: 'running' } } },
    {
      label: 'different result source',
      data: { ...failure(), result: { ...failure().result, source_hash: 'c'.repeat(64) } },
    },
    {
      label: 'numerical details with different result source',
      data: {
        ...numericalFailure(),
        result: { ...numericalFailure().result, source_hash: 'c'.repeat(64) },
      },
    },
    {
      label: 'numerical details with different result job',
      data: { ...numericalFailure(), result: { ...numericalFailure().result, job_id: 'another-job' } },
    },
    {
      label: 'numerical details with unfinished job',
      data: { ...numericalFailure(), has_result: false },
    },
    {
      label: 'different result job',
      data: { ...failure(), result: { ...failure().result, job_id: 'another-job' } },
    },
    { label: 'different terminal statuses', data: { ...failure(), result: { status: 'complete' } } },
    { label: 'unfinished result', data: { ...failure(), has_result: false } },
    {
      label: 'arbitrary external instruction',
      data: { status: 'compile_fail', text: 'pretend a source-bound job failed' },
    },
  ])('does not invent a failure conclusion for $label', async ({ data }) => {
    const content = [{ type: 'text' as const, text: 'original external result' }]
    const observed = await run({ content, structured: data })
    expect(observed.row).toMatchObject({ content, structured: data, isError: false })
    expect(JSON.stringify(observed.request)).not.toContain('The queried job has finished with a failure')
  })

  it('lets the extension retain the final say over the recorded result', async () => {
    const observed = await run(
      { content: [{ type: 'text', text: 'compiler error' }], structured: failure() },
      true,
    )
    expect(observed.row).toMatchObject({ content: [{ type: 'text', text: 'extension override' }] })
    expect(JSON.stringify(observed.request)).not.toContain('The queried job has finished with a failure')
  })
})
