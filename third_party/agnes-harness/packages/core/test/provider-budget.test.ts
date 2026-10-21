import type { ToolDef } from '@agnes/extension-api'
import type { InferenceEvent, ModelRecord, Provider, RequestBody } from '@agnes/protocol'
import { describe, expect, it } from 'vitest'
import { MemoryStorage } from '../src/log/memory-storage.js'
import { ToolRegistry } from '../src/registry/tools.js'
import { CompactionRunner } from '../src/step/compaction.js'
import { presetDefaults } from '../src/step/preset.js'
import { SessionImpl } from '../src/step/session.js'
import { sentFor, textTurn } from './helpers/fake-provider.js'
import { actor, openSession, readTool } from './helpers/open-session.js'

function wireProvider(attempts = 1, script?: (request: RequestBody) => InferenceEvent[]) {
  const requests: RequestBody[] = []
  const options: Array<Parameters<Provider['infer']>[1]> = []
  const dispatches: RequestBody[] = []
  const provider: Provider = {
    supportsAttemptAdmission: true,
    models: () => [],
    async *infer(request, input) {
      requests.push(request)
      options.push(input)
      for (let attempt = 0; attempt < attempts; attempt++) {
        if (input.beforeAttempt && !(await input.beforeAttempt({ signal: input.signal }))) {
          yield {
            type: 'error',
            reason: 'error',
            code: 'QUOTA',
            message: 'provider request limit reached',
            retryable: false,
          }
          return
        }
        dispatches.push(request)
      }
      for (const event of script?.(request) ?? textTurn('answer'))
        yield event.type === 'sent' ? sentFor(request, event) : event
    },
  }
  return { provider, requests, options, dispatches }
}

async function prompt(session: Awaited<ReturnType<typeof openSession>>['session']) {
  await session.enqueue('next-turn', { actor, content: [{ type: 'text', text: 'question' }] })
  return session.run({ until: 'turn-end', signal: new AbortController().signal })
}

async function consume(provider: Provider, request: RequestBody, signal = new AbortController().signal) {
  const events: InferenceEvent[] = []
  for await (const event of provider.infer(request, { signal, toolNames: [] })) events.push(event)
  return events
}

function kernelRegistry(
  names = ['kernel_get_task', 'kernel_submit', 'kernel_get_job'],
  ran = () => {},
  deferred = false,
) {
  const registry = new ToolRegistry()
  for (const name of names) {
    const tool = readTool(async () => {
      ran()
      return { content: [{ type: 'text' as const, text: 'executed' }] }
    }) as ToolDef
    registry.add(
      {
        ...tool,
        name,
        meta: { ...tool.meta, ...(deferred && name === 'kernel_get_job' ? { deferLoading: true } : {}) },
      },
      { source: 'agnes/kernel-test', trust: 'builtin' },
    )
  }
  return registry
}

function clippedKernelTurn(): InferenceEvent[] {
  return [
    ...textTurn('partial kernel source').filter((event) => event.type !== 'done'),
    {
      type: 'toolcall_end',
      call: { toolUseId: '', name: 'kernel_submit', args: {}, ordinal: 0 },
      via: 'native',
    },
    { type: 'done', reason: 'length' },
  ]
}

describe('durable session provider-request admission', () => {
  it('shows restored allowance in actual requests without charging the guidance read', async () => {
    const f = wireProvider()
    const { session, log } = await openSession({ provider: f.provider, maxProviderRequests: 24 })
    expect(session.providerRequestBudgetSnapshot()).toEqual({ cap: 24, used: 0, remaining: 24 })
    expect((await prompt(session)).reason).toBe('completed')
    expect(JSON.stringify(f.requests[0])).toContain(
      'Remaining provider requests including this dispatch: 24 of 24; already admitted: 0.',
    )
    expect((await prompt(session)).reason).toBe('completed')
    expect(JSON.stringify(f.requests[1])).toContain(
      'Remaining provider requests including this dispatch: 23 of 24; already admitted: 1.',
    )
    expect(session.providerRequestBudgetSnapshot()).toEqual({ cap: 24, used: 2, remaining: 22 })
    expect(await log.scan({ type: 'x/core/provider-request-admitted', limit: 100 })).toHaveLength(2)
    await session.close()
  })

  it('omits the guidance for an uncapped session', async () => {
    const f = wireProvider()
    const { session } = await openSession({ provider: f.provider })
    expect(session.providerRequestBudgetSnapshot()).toBeUndefined()
    expect((await prompt(session)).reason).toBe('completed')
    expect(JSON.stringify(f.requests[0])).not.toContain('Remaining provider requests')
    await session.close()
  })

  it('shows uncertain durable attempts after reopen without restoring spent allowance', async () => {
    const storage = new MemoryStorage()
    const f = wireProvider()
    const first = await openSession({ storage, provider: f.provider, maxProviderRequests: 24 })
    await prompt(first.session)
    expect(await f.options[0]?.beforeAttempt?.({ signal: new AbortController().signal })).toBe(true)
    await first.session.close()
    const reopened = await openSession({
      storage,
      provider: f.provider,
      writerRunId: 'r2',
      maxProviderRequests: 100,
    })
    expect(reopened.session.providerRequestBudgetSnapshot()).toEqual({ cap: 24, used: 2, remaining: 22 })
    await prompt(reopened.session)
    expect(JSON.stringify(f.requests.at(-1))).toContain(
      'Remaining provider requests including this dispatch: 22 of 24; already admitted: 2.',
    )
    expect(f.dispatches).toHaveLength(2)
    await reopened.session.close()
  })

  it('adds query guidance only at a low remaining allowance and still refuses the next dispatch', async () => {
    const f = wireProvider()
    const { session } = await openSession({ provider: f.provider, maxProviderRequests: 5 })
    await prompt(session)
    expect(JSON.stringify(f.requests[0])).not.toContain('avoid repeated queries')
    await prompt(session)
    expect(JSON.stringify(f.requests[1])).toContain('avoid repeated queries')
    for (let index = 0; index < 3; index++) await prompt(session)
    expect(session.providerRequestBudgetSnapshot()).toEqual({ cap: 5, used: 5, remaining: 0 })
    expect((await prompt(session)).reason).toBe('budget')
    expect(f.dispatches).toHaveLength(5)
    await session.close()
  })

  it('shares the explicit cap across turns without changing maxSteps', async () => {
    const f = wireProvider()
    const preset = presetDefaults()
    preset.budget.maxSteps = 50
    const { session, log } = await openSession({ provider: f.provider, preset, maxProviderRequests: 2 })
    expect((await prompt(session)).reason).toBe('completed')
    expect((await prompt(session)).reason).toBe('completed')
    expect((await prompt(session)).reason).toBe('budget')
    expect(f.dispatches).toHaveLength(2)
    expect(session.preset.budget.maxSteps).toBe(50)
    const rows = await log.scan({ type: 'x/core/provider-request-admitted', limit: 10 })
    expect(rows.map((row) => (row.data as { used: number }).used)).toEqual([1, 2])
    await session.close()
  })

  it.each([2, 24])(
    'bounds concurrent primary/compaction calls to %i before remote cancellation',
    async (cap) => {
      const f = wireProvider()
      const { session, log } = await openSession({ provider: f.provider, maxProviderRequests: cap })
      expect((await prompt(session)).reason).toBe('completed')
      const initial = f.requests[0] as RequestBody
      const results = await Promise.all(
        Array.from({ length: 40 }, (_, index) =>
          consume(session.d.provider, {
            ...initial,
            slot: index % 2 ? 'compaction' : 'primary',
            kind: 'summary',
          }),
        ),
      )
      expect(f.dispatches).toHaveLength(cap)
      expect(results.filter((events) => events.some((event) => event.type === 'error'))).toHaveLength(
        41 - cap,
      )
      expect(await log.scan({ type: 'x/core/provider-request-admitted', limit: 100 })).toHaveLength(cap)
      await session.close()
    },
  )

  it('charges every upstream retry within one provider invocation', async () => {
    const f = wireProvider(3)
    const { session, log } = await openSession({ provider: f.provider, maxProviderRequests: 2 })
    expect((await prompt(session)).reason).toBe('budget')
    expect(f.requests).toHaveLength(1)
    expect(f.dispatches).toHaveLength(2)
    expect(await log.scan({ type: 'x/core/provider-request-admitted', limit: 10 })).toHaveLength(2)
    await session.close()
  })

  it('restores uncertain admissions and the original cap after reopen', async () => {
    const storage = new MemoryStorage()
    const f = wireProvider()
    const first = await openSession({ storage, provider: f.provider, maxProviderRequests: 2 })
    await prompt(first.session)
    const request = f.requests[0] as RequestBody
    // A committed admission remains spent even if the process disappears before the wire answers.
    const options = f.options[0] as Parameters<Provider['infer']>[1]
    expect(await options.beforeAttempt?.({ signal: new AbortController().signal })).toBe(true)
    await first.session.close()
    const reopened = await openSession({
      storage,
      provider: f.provider,
      writerRunId: 'r2',
      maxProviderRequests: 100,
    })
    expect((await prompt(reopened.session)).reason).toBe('budget')
    expect((await consume(reopened.session.d.provider, request)).some((e) => e.type === 'error')).toBe(true)
    expect(f.dispatches).toHaveLength(1)
    await reopened.session.close()
  })

  it('does not remove a persisted cap when the reopened profile omits it', async () => {
    const storage = new MemoryStorage()
    const f = wireProvider()
    const first = await openSession({ storage, provider: f.provider, maxProviderRequests: 1 })
    await prompt(first.session)
    await first.session.close()
    const reopened = await openSession({ storage, provider: f.provider, writerRunId: 'r2' })
    expect((await prompt(reopened.session)).reason).toBe('budget')
    expect(f.dispatches).toHaveLength(1)
    await reopened.session.close()
  })

  it('does not wrap or charge the same provider twice when start is repeated', async () => {
    const f = wireProvider()
    const { session, log } = await openSession({ provider: f.provider, maxProviderRequests: 2 })
    await session.start()
    expect((await prompt(session)).reason).toBe('completed')
    expect(f.dispatches).toHaveLength(1)
    expect(await log.scan({ type: 'x/core/provider-request-admitted', limit: 10 })).toHaveLength(1)
    await session.close()
  })

  it('refuses enabling a cap beside an already active uncapped session writer', async () => {
    const f = wireProvider()
    const { session } = await openSession({ provider: f.provider })
    expect(() => new SessionImpl({ ...session.d, lane: 'aux', maxProviderRequests: 1 })).toThrow(
      'enabling a provider request cap requires reopening the session writer',
    )
    expect((await prompt(session)).reason).toBe('completed')
    expect(f.options[0]?.beforeAttempt).toBeUndefined()
    await session.close()
  })

  it('fails closed when a cap is newly applied to a legacy used session', async () => {
    const storage = new MemoryStorage()
    const f = wireProvider()
    const first = await openSession({ storage, provider: f.provider })
    await prompt(first.session)
    await first.session.close()
    const reopened = await openSession({
      storage,
      provider: f.provider,
      writerRunId: 'r2',
      maxProviderRequests: 24,
    })
    expect((await prompt(reopened.session)).reason).toBe('budget')
    expect(f.dispatches).toHaveLength(1)
    await reopened.session.close()
  })

  it('does not invoke a provider that cannot enforce attempt admission', async () => {
    const f = wireProvider()
    const { supportsAttemptAdmission: _capability, ...unsupported } = f.provider
    const { session } = await openSession({ provider: unsupported, maxProviderRequests: 24 })
    expect((await prompt(session)).reason).toBe('budget')
    expect(f.requests).toHaveLength(0)
    expect(f.dispatches).toHaveLength(0)
    await session.close()
  })

  it('does not install an attempt callback without an explicit or persisted cap', async () => {
    const f = wireProvider(3)
    const { session, log } = await openSession({ provider: f.provider })
    expect((await prompt(session)).reason).toBe('completed')
    expect(f.dispatches).toHaveLength(3)
    expect(f.options[0]).not.toHaveProperty('beforeAttempt')
    expect(await log.scan({ type: 'x/core/provider-request-admitted', limit: 10 })).toEqual([])
    await session.close()
  })

  it('refuses cancelled work without charging a new slot', async () => {
    const f = wireProvider()
    const { session, log } = await openSession({ provider: f.provider, maxProviderRequests: 24 })
    await prompt(session)
    const abort = new AbortController()
    abort.abort()
    await consume(session.d.provider, f.requests[0] as RequestBody, abort.signal)
    expect(f.dispatches).toHaveLength(1)
    expect(await log.scan({ type: 'x/core/provider-request-admitted', limit: 100 })).toHaveLength(1)
    await session.close()
  })

  it('can admit while a caller owns the session phase lock', async () => {
    const f = wireProvider()
    const { session } = await openSession({ provider: f.provider, maxProviderRequests: 24 })
    await prompt(session)
    await session.locked(() => consume(session.d.provider, f.requests[0] as RequestBody))
    expect(f.dispatches).toHaveLength(2)
    await session.close()
  })

  it('shares 24 slots with real parallel split compaction and its length retries', async () => {
    const f = wireProvider(1, (request) =>
      request.kind === 'summary'
        ? [
            ...textTurn('').filter((event) => event.type !== 'done'),
            { type: 'done' as const, reason: 'length' as const },
          ]
        : textTurn('old-answer-summarized-away').map((event) =>
            event.type === 'usage'
              ? { ...event, tokens: { ...event.tokens, input: 1000, output: 100 } }
              : event,
          ),
    )
    let primaryWindow = 200000
    const model = (
      id: string,
      slot: NonNullable<ModelRecord['slot']>,
      contextWindow: number,
    ): ModelRecord => ({
      id,
      name: id,
      slot,
      route: 'default',
      api: 'openai-completions',
      baseUrl: 'https://example.invalid/v1',
      reasoning: false,
      input: ['text'],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow,
      maxTokens: 4096,
      toolCallFormats: ['native'],
      thinkingReplay: 'native',
      contract_id: null,
    })
    f.provider.models = () => [
      model('answer-model', 'primary', primaryWindow),
      model('summary-model', 'compaction', 1000),
    ]
    const { session, log } = await openSession({ provider: f.provider, maxProviderRequests: 24 })
    for (let turn = 0; turn < 22; turn++) expect((await prompt(session)).reason).toBe('completed')
    primaryWindow = 100
    delete session.preset.model.contextWindow
    session.preset.model.id.compaction = 'summary-model'
    session.preset.compaction.reserveTokens = 80
    session.compaction = new CompactionRunner({
      plan: async (payload) => {
        const nodes = payload.getSurface()
        const [a, b, c, d] = nodes
        const kept = nodes.at(-1)
        if (!a || !b || !c || !d || !kept) throw new Error('missing split history')
        return {
          keepFromSeq: kept.seq,
          summarizeRange: [a.seq, b.seq],
          turnPrefixRange: [c.seq, d.seq],
          prompts: { system: 'S', history: 'summarize', prefix: 'summarize prefix' },
          maxTokens: 100,
          details: { readFiles: [], modifiedFiles: [] },
        }
      },
      onCompact: async () => undefined,
    })
    expect((await prompt(session)).reason).toBe('budget')
    expect(f.dispatches).toHaveLength(24)
    expect(f.dispatches.slice(22).map((request) => request.slot)).toEqual(['compaction', 'compaction'])
    expect(f.requests.filter((request) => request.kind === 'summary')).toHaveLength(4)
    expect(await log.scan({ type: 'x/core/provider-request-admitted', limit: 100 })).toHaveLength(24)
    await session.close()
  })

  async function requestTemplate() {
    const f = wireProvider()
    const { session } = await openSession({ provider: f.provider, key: 'template' })
    await prompt(session)
    await session.close()
    return f.requests[0] as RequestBody
  }

  it.each([1, 2])('shares cap %i across distinct SessionImpl lanes using the same log', async (cap) => {
    const request = await requestTemplate()
    const f = wireProvider()
    const { session, log } = await openSession({ provider: f.provider, maxProviderRequests: cap })
    const aux = new SessionImpl({ ...session.d, provider: f.provider, lane: 'aux', maxProviderRequests: cap })
    await aux.start()
    await Promise.all(
      Array.from({ length: 8 }, (_, index) =>
        consume(index % 2 ? aux.d.provider : session.d.provider, {
          ...request,
          kind: 'summary',
          slot: 'compaction',
        }),
      ),
    )
    expect(f.dispatches).toHaveLength(cap)
    const admissions = await log.scan({ type: 'x/core/provider-request-admitted', limit: 100 })
    expect(admissions.map((event) => (event.data as { used: number }).used)).toEqual(
      Array.from({ length: cap }, (_, index) => index + 1),
    )
    await session.close()
  })

  it('fences independent logs even if two instances reuse the same writer identity', async () => {
    const request = await requestTemplate()
    const storage = new MemoryStorage()
    const f = wireProvider()
    const a = await openSession({ storage, provider: f.provider, maxProviderRequests: 1 })
    const b = await openSession({ storage, provider: f.provider, maxProviderRequests: 1 })
    const attempts = await Promise.allSettled([
      consume(a.session.d.provider, request),
      consume(b.session.d.provider, request),
    ])
    expect(f.dispatches).toHaveLength(1)
    expect(attempts.filter((attempt) => attempt.status === 'rejected')).toHaveLength(1)
    await a.session.close()
    await b.session.close()
  })

  it('keeps a slot when cancellation lands after its durable commit but before HTTP', async () => {
    const request = await requestTemplate()
    const storage = new MemoryStorage()
    const abort = new AbortController()
    const commit = storage.commit.bind(storage)
    storage.commit = async (key, tx) => {
      const result = await commit(key, tx)
      if (tx.events.some((event) => event.type === 'x/core/provider-request-admitted')) abort.abort()
      return result
    }
    const f = wireProvider()
    const { session, log } = await openSession({ storage, provider: f.provider, maxProviderRequests: 1 })
    await consume(session.d.provider, request, abort.signal)
    await consume(session.d.provider, request)
    expect(f.dispatches).toHaveLength(0)
    expect(await log.scan({ type: 'x/core/provider-request-admitted', limit: 100 })).toHaveLength(1)
    await session.close()
  })

  it('keeps usage when the reopened profile tightens its cap below previous admissions', async () => {
    const storage = new MemoryStorage()
    const f = wireProvider()
    const first = await openSession({ storage, provider: f.provider, maxProviderRequests: 3 })
    await prompt(first.session)
    await prompt(first.session)
    await first.session.close()
    const reopened = await openSession({
      storage,
      provider: f.provider,
      writerRunId: 'r2',
      maxProviderRequests: 1,
    })
    expect((await prompt(reopened.session)).reason).toBe('budget')
    expect(f.dispatches).toHaveLength(2)
    const rows = await reopened.log.scan({ type: 'x/core/provider-request-budget', order: 'desc', limit: 1 })
    expect(rows[0]?.data).toMatchObject({ cap: 1, used: 2 })
    await reopened.session.close()
  })

  it('ignores untrusted budget lookalikes when restoring a spent cap', async () => {
    const storage = new MemoryStorage()
    const f = wireProvider()
    const first = await openSession({ storage, provider: f.provider, maxProviderRequests: 1 })
    await prompt(first.session)
    await first.session.append([
      first.session.ev(
        'x/core/provider-request-admitted',
        {
          sessionKey: first.session.key,
          cap: 100,
          used: 0,
        },
        { ignorable: true, origin: 'ext:rogue', trust: 'untrusted' },
      ),
    ])
    await first.session.close()
    const reopened = await openSession({ storage, provider: f.provider, writerRunId: 'r2' })
    expect((await prompt(reopened.session)).reason).toBe('budget')
    expect(f.dispatches).toHaveLength(1)
    await reopened.session.close()
  })

  it.each([0, -1, 1.5, Number.POSITIVE_INFINITY, Number.MAX_SAFE_INTEGER + 1])(
    'rejects invalid explicit cap %s',
    async (maxProviderRequests) => {
      await expect(
        openSession({ provider: wireProvider().provider, maxProviderRequests }),
      ).rejects.toMatchObject({ code: 'E_ENVELOPE' })
    },
  )
  it('continues one clipped native kernel response within the same physical allowance without executing its calls', async () => {
    let response = 0
    let ran = 0
    const f = wireProvider(1, () => (response++ === 0 ? clippedKernelTurn() : textTurn('recovered')))
    const { session, log } = await openSession({
      provider: f.provider,
      registry: kernelRegistry(undefined, () => ran++),
      maxProviderRequests: 24,
    })
    expect((await prompt(session)).reason).toBe('completed')
    expect(f.dispatches).toHaveLength(2)
    expect(ran).toBe(0)
    expect(await log.scan({ type: 'tool/call', limit: 10 })).toHaveLength(0)
    expect(await log.scan({ type: 'x/core/output-limit-continuation', limit: 10 })).toHaveLength(1)
    expect(session.providerRequestBudgetSnapshot()).toEqual({ cap: 24, used: 2, remaining: 22 })
    expect(f.requests[1]?.model).toEqual(f.requests[0]?.model)
    expect(JSON.stringify(f.requests[1])).toContain('shorter complete tool calls')
    await session.close()
  })

  it('terminates the second clipped response in a native kernel turn and retains both charges', async () => {
    const f = wireProvider(1, clippedKernelTurn)
    const { session, log } = await openSession({
      provider: f.provider,
      registry: kernelRegistry(),
      maxProviderRequests: 24,
    })
    expect(await prompt(session)).toMatchObject({ reason: 'error', error: { code: 'OUTPUT_LIMIT' } })
    expect(f.dispatches).toHaveLength(2)
    expect(await log.scan({ type: 'x/core/output-limit-continuation', limit: 10 })).toHaveLength(1)
    expect(await log.scan({ type: 'tool/call', limit: 10 })).toHaveLength(0)
    expect(session.providerRequestBudgetSnapshot()).toEqual({ cap: 24, used: 2, remaining: 22 })
    await session.close()
  })

  it.each([1, undefined])(
    'retains native truncation termination without available explicit allowance (%s)',
    async (cap) => {
      const f = wireProvider(1, clippedKernelTurn)
      const { session, log } = await openSession({
        provider: f.provider,
        registry: kernelRegistry(),
        ...(cap === undefined ? {} : { maxProviderRequests: cap }),
      })
      expect(await prompt(session)).toMatchObject({ reason: 'error', error: { code: 'OUTPUT_LIMIT' } })
      expect(f.dispatches).toHaveLength(1)
      expect(await log.scan({ type: 'x/core/output-limit-continuation', limit: 10 })).toHaveLength(0)
      expect(await log.scan({ type: 'tool/call', limit: 10 })).toHaveLength(0)
      await session.close()
    },
  )

  it.each([['kernel_get_task'], ['kernel_get_task', 'kernel_submit']])(
    'retains truncation termination for incomplete native disclosure (%j)',
    async (...names) => {
      const f = wireProvider(1, clippedKernelTurn)
      const { session, log } = await openSession({
        provider: f.provider,
        registry: kernelRegistry(names),
        maxProviderRequests: 24,
      })
      expect(await prompt(session)).toMatchObject({ reason: 'error', error: { code: 'OUTPUT_LIMIT' } })
      expect(f.dispatches).toHaveLength(1)
      expect(await log.scan({ type: 'x/core/output-limit-continuation', limit: 10 })).toHaveLength(0)
      await session.close()
    },
  )

  it('replays the one clipped-response continuation after reopen without granting a second recovery', async () => {
    const storage = new MemoryStorage()
    const f = wireProvider(1, clippedKernelTurn)
    const first = await openSession({
      storage,
      provider: f.provider,
      registry: kernelRegistry(),
      maxProviderRequests: 24,
    })
    await first.session.enqueue('next-turn', { actor, content: [{ type: 'text', text: 'kernel task' }] })
    await first.session.acceptInput()
    expect(await first.session.runInference()).toEqual({ phase: 'checkpoint' })
    expect(await first.log.scan({ type: 'x/core/output-limit-continuation', limit: 10 })).toHaveLength(1)
    await first.session.close()
    const second = await openSession({
      storage,
      writerRunId: 'r2',
      provider: f.provider,
      registry: kernelRegistry(),
      maxProviderRequests: 24,
    })
    expect(
      await second.session.run({ until: 'turn-end', signal: new AbortController().signal }),
    ).toMatchObject({ reason: 'error', error: { code: 'OUTPUT_LIMIT' } })
    expect(f.dispatches).toHaveLength(2)
    expect(await second.log.scan({ type: 'x/core/output-limit-continuation', limit: 10 })).toHaveLength(1)
    expect(second.session.providerRequestBudgetSnapshot()).toEqual({ cap: 24, used: 2, remaining: 22 })
    await second.session.close()
  })

  it('does not recover a clipped response after cancellation during inference', async () => {
    let current: SessionImpl
    const f = wireProvider(1, () => {
      current.ac.abort()
      return clippedKernelTurn()
    })
    const { session, log } = await openSession({
      provider: f.provider,
      registry: kernelRegistry(),
      maxProviderRequests: 24,
    })
    current = session
    expect((await prompt(session)).reason).toBe('aborted')
    expect(f.dispatches).toHaveLength(1)
    expect(await log.scan({ type: 'x/core/output-limit-continuation', limit: 10 })).toHaveLength(0)
    await session.close()
  })

  it.each([
    { origin: 'ext:rogue', trust: 'untrusted', lane: 'main', recover: true },
    { origin: 'system', trust: 'trusted', lane: 'main', recover: false },
    { origin: 'system', trust: 'trusted', lane: 'other', recover: true },
  ] as const)(
    'handles a mismatched continuation marker from $origin/$trust/$lane',
    async ({ origin, trust, lane, recover }) => {
      let response = 0
      const f = wireProvider(1, () => (response++ === 0 ? clippedKernelTurn() : textTurn('recovered')))
      const { session, log } = await openSession({
        provider: f.provider,
        registry: kernelRegistry(),
        maxProviderRequests: 24,
      })
      await session.enqueue('next-turn', { actor, content: [{ type: 'text', text: 'kernel task' }] })
      await session.acceptInput()
      await log.append([
        {
          ...session.ev(
            'x/core/output-limit-continuation',
            { schemaVersion: 1, scope: '0'.repeat(64) },
            { ignorable: true, origin, trust },
          ),
          lane,
        },
      ])
      const outcome = await session.run({ until: 'turn-end', signal: new AbortController().signal })
      expect(outcome.reason).toBe(recover ? 'completed' : 'error')
      if (!recover) expect(outcome.error?.code).toBe('OUTPUT_LIMIT')
      expect(f.dispatches).toHaveLength(recover ? 2 : 1)
      expect(await log.scan({ type: 'tool/call', limit: 10 })).toHaveLength(0)
      await session.close()
    },
  )

  it('allows a new turn one recovery while retaining the cumulative allowance', async () => {
    let response = 0
    const f = wireProvider(1, () => (response++ % 2 === 0 ? clippedKernelTurn() : textTurn('recovered')))
    const { session, log } = await openSession({
      provider: f.provider,
      registry: kernelRegistry(),
      maxProviderRequests: 24,
    })
    expect((await prompt(session)).reason).toBe('completed')
    expect((await prompt(session)).reason).toBe('completed')
    expect(f.dispatches).toHaveLength(4)
    const markers = await log.scan({ type: 'x/core/output-limit-continuation', limit: 10 })
    expect(markers).toHaveLength(2)
    expect(markers[0]?.data).not.toEqual(markers[1]?.data)
    expect(session.providerRequestBudgetSnapshot()).toEqual({ cap: 24, used: 4, remaining: 20 })
    await session.close()
  })

  it('retains truncation termination when registered kernel tools are not disclosed', async () => {
    const f = wireProvider(1, clippedKernelTurn)
    const { session, log } = await openSession({
      provider: f.provider,
      registry: kernelRegistry(undefined, undefined, true),
      maxProviderRequests: 24,
    })
    expect(await prompt(session)).toMatchObject({ reason: 'error', error: { code: 'OUTPUT_LIMIT' } })
    expect(f.dispatches).toHaveLength(1)
    expect(await log.scan({ type: 'x/core/output-limit-continuation', limit: 10 })).toHaveLength(0)
    await session.close()
  })

  it.each(['cancel', 'spend'] as const)(
    'checks the live cancellation and allowance after reading recovery history (%s)',
    async (duringRead) => {
      const f = wireProvider(1, clippedKernelTurn)
      const { session, log } = await openSession({
        provider: f.provider,
        registry: kernelRegistry(),
        maxProviderRequests: duringRead === 'spend' ? 2 : 24,
      })
      const scan = log.scan.bind(log)
      let armed = true
      log.scan = async (query) => {
        const rows = await scan(query)
        if (armed && query?.type === 'x/core/output-limit-continuation') {
          armed = false
          if (duringRead === 'cancel') await session.abort()
          else await consume(session.d.provider, f.requests[0] as RequestBody)
        }
        return rows
      }
      const outcome = await prompt(session)
      expect(outcome.reason).toBe(duringRead === 'cancel' ? 'aborted' : 'error')
      if (duringRead === 'spend') expect(outcome.error?.code).toBe('OUTPUT_LIMIT')
      expect(await log.scan({ type: 'x/core/output-limit-continuation', limit: 10 })).toHaveLength(0)
      expect(f.dispatches).toHaveLength(duringRead === 'cancel' ? 1 : 2)
      expect(await log.scan({ type: 'tool/call', limit: 10 })).toHaveLength(0)
      await session.close()
    },
  )
})
