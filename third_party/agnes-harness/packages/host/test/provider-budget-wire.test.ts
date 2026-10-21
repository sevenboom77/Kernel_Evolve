import { mkdtempSync, realpathSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createProvider, NullContractStore, PiAdapter } from '@agnes/ai'
import { CompactionRunner } from '@agnes/core'
import type { ModelRecord } from '@agnes/protocol'
import { expect, it, vi } from 'vitest'
import { createTestHost } from '../testkit/index.js'

function answer(model: string, summary: boolean): Response {
  const events = [
    {
      type: 'message_start',
      message: {
        id: 'fixture',
        type: 'message',
        role: 'assistant',
        content: [],
        model,
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: 1000, output_tokens: 0 },
      },
    },
    { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
    {
      type: 'content_block_delta',
      index: 0,
      delta: { type: 'text_delta', text: summary ? 'cutoff summary' : 'old-answer-summarized-away' },
    },
    { type: 'content_block_stop', index: 0 },
    {
      type: 'message_delta',
      delta: { stop_reason: summary ? 'max_tokens' : 'end_turn', stop_sequence: null },
      usage: { output_tokens: summary ? 100 : 10 },
    },
    { type: 'message_stop' },
  ]
  return new Response(
    events.map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`).join(''),
    { status: 200, headers: { 'content-type': 'text/event-stream' } },
  )
}

it('keeps physical SDK dispatches within 24 across main turns, parallel compaction and retries', async () => {
  const root = realpathSync.native(mkdtempSync(join(tmpdir(), 'agnes-provider-budget-wire-')))
  let primaryWindow = 200000
  const model = (id: string, slot: NonNullable<ModelRecord['slot']>, contextWindow: number): ModelRecord => ({
    id,
    name: id,
    slot,
    route: 'bounded',
    api: 'anthropic-messages',
    baseUrl: 'https://admission.invalid',
    reasoning: id === 's',
    ...(id === 's' ? { thinkingLevelMap: { low: 'low' } } : {}),
    input: ['text'],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow,
    maxTokens: 4096,
    toolCallFormats: ['native'],
    thinkingReplay: 'native',
    contract_id: null,
  })
  const route = {
    route: 'bounded',
    api: 'anthropic-messages',
    baseUrl: 'https://admission.invalid',
    credentialRef: 'fixture',
    models: [model('m', 'primary', 200000), model('s', 'compaction', 1000)],
  }
  const calls: string[] = []
  vi.stubGlobal('fetch', async (input: RequestInfo | URL, init?: RequestInit) => {
    const request = new Request(input, init)
    const body = (await request.json()) as { model: string }
    calls.push(body.model)
    return answer(body.model, body.model === 's')
  })
  const adapter = new PiAdapter({ manualRoutes: [route], maxRetries: 2, sleep: async () => {} })
  const streams = vi.spyOn(adapter, 'stream')
  const provider = createProvider({
    adapters: [adapter],
    routes: { primary: { route: 'bounded', model: 'm' }, compaction: { route: 'bounded', model: 's' } },
    contract: new NullContractStore(),
    secrets: () => 'fixture-key',
    clock: () => 0,
  })
  provider.models = () => [model('m', 'primary', primaryWindow), model('s', 'compaction', 1000)]
  const { host } = await createTestHost({
    dataDir: root,
    provider,
    profileInputs: {
      user: {
        name: 'bounded-test',
        limits: { 'session.max_provider_requests': 24 },
        provider: { package: '@agnes/ai', adapters: ['@agnes/ai'], routes: [route] },
      },
    },
  })
  try {
    const session = await host.createSession({ key: 'bounded-wire', cwd: root })
    const prompt = async () => {
      await session.enqueue('next-turn', {
        actor: session.d.actor,
        content: [{ type: 'text', text: 'question' }],
      })
      return session.run({ until: 'turn-end', signal: new AbortController().signal })
    }
    for (let turn = 0; turn < 22; turn++) expect((await prompt()).reason).toBe('completed')
    primaryWindow = 100
    delete session.preset.model.contextWindow
    session.preset.model.id.compaction = 's'
    session.preset.model.route.compaction = 'bounded'
    session.preset.compaction.reserveTokens = 80
    session.compaction = new CompactionRunner({
      plan: async (payload) => {
        const [a, b, c, d] = payload.getSurface()
        const kept = payload.getSurface().at(-1)
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
      onCompact: async () => {},
    })
    const outcome = await prompt()
    expect(outcome.reason).toBe('budget')
    expect(calls).toEqual([...Array.from({ length: 22 }, () => 'm'), 's', 's'])
    const summaries = streams.mock.calls
      .map((call) => call[1])
      .filter((request) => request.kind === 'summary')
    expect(summaries).toHaveLength(4)
    expect(summaries.filter((request) => request.sampling?.thinking === 'low')).toHaveLength(2)
    const admissions = await session.scan({ type: 'x/core/provider-request-admitted', limit: 100 })
    expect(admissions).toHaveLength(24)
    expect(admissions.map((event) => (event.data as { used: number }).used)).toEqual(
      Array.from({ length: 24 }, (_, index) => index + 1),
    )
    const titleCalls = (await session.scan({ type: 'cost/ledger', limit: 100 })).filter(
      (event) => (event.data as { purpose?: string }).purpose === 'title',
    )
    expect(titleCalls).toEqual([])
    expect(await session.scan({ type: 'request/sent', limit: 100 })).toHaveLength(22)
    expect((await prompt()).reason).toBe('budget')
    expect(calls).toHaveLength(24)
  } finally {
    await host.close()
    vi.unstubAllGlobals()
    rmSync(root, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 })
  }
}, 30_000)
