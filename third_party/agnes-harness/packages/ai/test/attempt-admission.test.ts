import { createServer } from 'node:http'
import type { Provider } from '@agnes/protocol'
import type { AssistantMessage, AssistantMessageEvent } from '@earendil-works/pi-ai'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { createProvider, MediaAdapter, NullContractStore, PiAdapter } from '../src/index.js'
import { FakeAdapter, fakeModel, fakeRequest } from '../testkit/index.js'
import { assertLoopbackOnly, installLoopbackOnly, restoreLoopbackOnly } from './loopback-only.js'

const route = {
  route: 'bounded',
  api: 'anthropic-messages',
  baseUrl: 'https://admission.invalid',
  credentialRef: 'fixture',
  models: [
    fakeModel({ id: 'm', route: 'bounded', api: 'anthropic-messages', baseUrl: 'https://admission.invalid' }),
  ],
}
const request = () => fakeRequest({ route: 'bounded', model: 'm' })
const options = (signal = new AbortController().signal) => ({ signal, toolNames: [] })
const quota = {
  type: 'error',
  reason: 'error',
  code: 'QUOTA',
  message: 'provider request limit reached',
  retryable: false,
}

async function collect<T>(events: AsyncIterable<T>): Promise<T[]> {
  const out: T[] = []
  for await (const event of events) out.push(event)
  return out
}

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

function ok(): Response {
  const events = [
    {
      type: 'message_start',
      message: {
        id: 'fixture',
        type: 'message',
        role: 'assistant',
        content: [],
        model: 'm',
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: 1, output_tokens: 0 },
      },
    },
    { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
    { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'ok' } },
    { type: 'content_block_stop', index: 0 },
    {
      type: 'message_delta',
      delta: { stop_reason: 'end_turn', stop_sequence: null },
      usage: { output_tokens: 1 },
    },
    { type: 'message_stop' },
  ]
  return new Response(
    events.map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`).join(''),
    { status: 200, headers: { 'content-type': 'text/event-stream' } },
  )
}

function wire(statuses: number[] = []) {
  const calls: Request[] = []
  vi.stubGlobal('fetch', async (input: RequestInfo | URL, init?: RequestInit) => {
    calls.push(new Request(input, init))
    const status = statuses[calls.length - 1] ?? 200
    return status === 200
      ? ok()
      : new Response(JSON.stringify({ error: { message: 'fixture rejection' } }), {
          status,
          headers: { 'content-type': 'application/json' },
        })
  })
  return calls
}

function provider(cfg: Partial<ConstructorParameters<typeof PiAdapter>[0]> = {}) {
  const adapter = new PiAdapter({ manualRoutes: [route], maxRetries: 2, sleep: async () => {}, ...cfg })
  return createProvider({
    adapters: [adapter],
    routes: { primary: { route: 'bounded', model: 'm' } },
    contract: new NullContractStore(),
    secrets: () => 'fixture-key',
    clock: () => 0,
  })
}

beforeEach(() => installLoopbackOnly())
afterEach(() => {
  vi.unstubAllGlobals()
  restoreLoopbackOnly()
  assertLoopbackOnly()
})

it('rejects the first admission without HTTP, a sent stamp or retries', async () => {
  const calls = wire()
  const gate = vi.fn(async () => false)
  const events = await collect(provider().infer(request(), { ...options(), beforeAttempt: gate }))
  expect(calls).toHaveLength(0)
  expect(gate).toHaveBeenCalledTimes(1)
  expect(events).toEqual([quota])
})

it.each([
  'openai-completions',
  'openai-responses',
  'openai-codex-responses',
  'azure-openai-responses',
  'anthropic-messages',
  'mistral-conversations',
  'pi-messages',
])('%s exposes its actual SDK dispatch to admission', async (api) => {
  const calls = wire()
  const claims = Buffer.from(
    JSON.stringify({ 'https://api.openai.com/auth': { chatgpt_account_id: 'fixture' } }),
  ).toString('base64url')
  const p = provider({
    manualRoutes: [{ ...route, api }],
  })
  if (api === 'openai-codex-responses')
    p.registry.lookup('bounded')?.adapter.bindCredential('bounded', `x.${claims}.x`)
  const gate = vi.fn(async () => false)
  const events = await collect(p.infer(request(), { ...options(), beforeAttempt: gate }))
  expect(calls).toHaveLength(0)
  expect(gate).toHaveBeenCalledTimes(1)
  expect(events).toEqual([quota])
})

it('sanitizes admission exceptions without sending or retrying', async () => {
  const calls = wire()
  const gate = vi.fn(async () => {
    throw Object.assign(new Error('private callback material'), { code: 'BUDGET_EXCEEDED' })
  })
  const events = await collect(provider().infer(request(), { ...options(), beforeAttempt: gate }))
  expect(calls).toHaveLength(0)
  expect(gate).toHaveBeenCalledTimes(1)
  expect(events).toEqual([quota])
  expect(JSON.stringify(events)).not.toContain('private callback material')
})

it('awaits durable admission before the actual Anthropic SDK fetch', async () => {
  const calls = wire()
  const release = deferred<boolean>()
  const entered = deferred<void>()
  const gate = vi.fn(async ({ signal }: { signal: AbortSignal }) => {
    expect(signal).toBeInstanceOf(AbortSignal)
    entered.resolve()
    return release.promise
  })
  const result = collect(provider().infer(request(), { ...options(), beforeAttempt: gate }))
  // The old implementation never calls the callback; polling the known network boundary lets
  // this fail as an assertion rather than hanging on a callback that has not been implemented.
  await vi.waitFor(() => expect(gate).toHaveBeenCalledTimes(1))
  await entered.promise
  expect(calls).toHaveLength(0)
  release.resolve(true)
  const events = await result
  expect(calls).toHaveLength(1)
  expect(events[0]?.type).toBe('sent')
  expect(events.at(-1)).toEqual({ type: 'done', reason: 'stop' })
})

it('gates ordinary retries individually and retains the real first sent stamp', async () => {
  const calls = wire([503, 200])
  let available = 1
  const gate = vi.fn(async () => available-- > 0)
  const events = await collect(provider().infer(request(), { ...options(), beforeAttempt: gate }))
  expect(calls).toHaveLength(1)
  expect(gate).toHaveBeenCalledTimes(2)
  expect(events.filter((event) => event.type === 'sent')).toHaveLength(1)
  expect(events.at(-1)).toEqual(quota)
})

it('gates AUTH recovery even when ordinary retries are disabled', async () => {
  const calls = wire([401, 200])
  let available = 1
  const gate = vi.fn(async () => available-- > 0)
  const recovered = vi.fn(async () => true)
  const p = provider({ resolveCredential: async () => 'fixture-key', recoverRejectedAuth: recovered })
  const events = await collect(p.infer(request(), { ...options(), retry: false, beforeAttempt: gate }))
  expect(recovered).toHaveBeenCalledTimes(1)
  expect(gate).toHaveBeenCalledTimes(2)
  expect(calls).toHaveLength(1)
  expect(events.at(-1)).toEqual(quota)
})

it('caps twenty concurrent streams at three admitted physical requests', async () => {
  const calls = wire()
  const p = provider()
  let available = 3
  const gate = vi.fn(async (input: { signal: AbortSignal }) => {
    expect(Object.keys(input)).toEqual(['signal'])
    const admitted = available-- > 0
    await Promise.resolve()
    return admitted
  })
  const episodes = await Promise.all(
    Array.from({ length: 20 }, () => collect(p.infer(request(), { ...options(), beforeAttempt: gate }))),
  )
  expect(calls).toHaveLength(3)
  expect(gate).toHaveBeenCalledTimes(20)
  expect(episodes.filter((events) => events.at(-1)?.type === 'done')).toHaveLength(3)
  expect(episodes.filter((events) => events.length === 1 && events[0]?.type === 'error')).toHaveLength(17)
})

it('never dispatches a late admission after the caller has cancelled', async () => {
  const calls = wire()
  const release = deferred<boolean>()
  const gate = vi.fn(async () => release.promise)
  const stop = new AbortController()
  const result = collect(provider().infer(request(), { ...options(stop.signal), beforeAttempt: gate }))
  await vi.waitFor(() => expect(gate).toHaveBeenCalledTimes(1))
  stop.abort()
  release.resolve(true)
  const events = await result
  await Promise.resolve()
  expect(calls).toHaveLength(0)
  expect(events.at(-1)).toMatchObject({ type: 'error', code: 'ABORTED', retryable: false })
  expect(events.filter((event) => event.type === 'sent')).toHaveLength(0)
})

it('keeps concurrent sessions admission closures separate on the same route and credential', async () => {
  const calls = wire()
  const p = provider()
  const denied = vi.fn(async () => false)
  const allowed = vi.fn(async () => true)
  const forSession = (sessionKey: string) =>
    fakeRequest({
      route: 'bounded',
      model: 'm',
      sessionKey,
      messages: [{ role: 'user', content: [{ type: 'text', text: sessionKey }] }],
    })
  const [blocked, succeeded] = await Promise.all([
    collect(p.infer(forSession('blocked-session'), { ...options(), beforeAttempt: denied })),
    collect(p.infer(forSession('allowed-session'), { ...options(), beforeAttempt: allowed })),
  ])
  expect(denied).toHaveBeenCalledTimes(1)
  expect(allowed).toHaveBeenCalledTimes(1)
  expect(calls).toHaveLength(1)
  expect(blocked).toEqual([quota])
  expect(succeeded.at(-1)).toEqual({ type: 'done', reason: 'stop' })
  expect(await calls[0]?.text()).toContain('allowed-session')
})

it('preserves normal retry behavior when admission is absent', async () => {
  const calls = wire([503, 200])
  const events = await collect(provider().infer(request(), options()))
  expect(calls).toHaveLength(2)
  expect(events[0]?.type).toBe('sent')
  expect(events.at(-1)).toEqual({ type: 'done', reason: 'stop' })
})

it('prevents native fetch redirect resends from bypassing admission', async () => {
  installLoopbackOnly()
  const paths: string[] = []
  const server = createServer(async (req, res) => {
    for await (const _ of req) {
    }
    paths.push(req.url ?? '')
    if (req.url?.startsWith('/v1/messages')) {
      res.writeHead(307, { location: '/redirected' })
      res.end()
    } else {
      const response = ok()
      res.writeHead(200, { 'content-type': 'text/event-stream' })
      res.end(await response.text())
    }
  })
  try {
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
    const address = server.address()
    if (!address || typeof address === 'string') throw new Error('missing loopback port')
    const p = provider({
      manualRoutes: [{ ...route, baseUrl: `http://127.0.0.1:${address.port}` }],
    })
    let available = 1
    const gate = vi.fn(async () => available-- > 0)
    const events = await collect(p.infer(request(), { ...options(), beforeAttempt: gate }))
    expect(paths).toHaveLength(1)
    expect(paths[0]).toMatch(/^\/v1\/messages/)
    expect(gate).toHaveBeenCalledTimes(1)
    expect(events.at(-1)?.type).toBe('error')
  } finally {
    server.closeAllConnections()
    await new Promise<void>((resolve) => server.close(() => resolve()))
    restoreLoopbackOnly()
    assertLoopbackOnly()
  }
})

it('does not re-admit or send if a wire library catches denial and retries internally', async () => {
  const calls = wire()
  let available = 1
  const gate = vi.fn(async () => available-- > 0)
  const p = provider({
    streamImpl: (_model, _context, opts) =>
      (async function* (): AsyncIterable<AssistantMessageEvent> {
        for (let attempt = 0; attempt < 3; attempt++) {
          try {
            await opts?.fetch?.('https://admission.invalid/v1/messages', {
              method: 'POST',
              body: JSON.stringify({ attempt }),
            })
          } catch {
            // Simulate a vendor SDK treating all fetch exceptions as internally retryable.
          }
        }
        yield { type: 'done', reason: 'stop', message: {} as AssistantMessage }
      })(),
  })
  const events = await collect(p.infer(request(), { ...options(), beforeAttempt: gate }))
  expect(calls).toHaveLength(1)
  expect(gate).toHaveBeenCalledTimes(2)
  expect(events.at(-1)).toEqual(quota)
})

it('sanitizes a callback exception when PiAdapter is called directly', async () => {
  const calls = wire()
  const adapter = new PiAdapter({ manualRoutes: [route], sleep: async () => {} })
  adapter.bindCredential('bounded', 'fixture-key')
  const gate = vi.fn(async () => {
    throw new Error('private direct callback material')
  })
  const events = await collect(
    adapter.stream('bounded', request(), {
      ...options(),
      beforeAttempt: gate,
      sessionKey: 'fixture',
      timeoutMs: { firstToken: 1000, total: 5000 },
    }),
  )
  expect(calls).toHaveLength(0)
  expect(gate).toHaveBeenCalledTimes(1)
  expect(events).toEqual([quota])
})

it('fails closed before entering a custom adapter without admission capability', async () => {
  const adapter = new FakeAdapter({ id: 'opaque', routes: [route], models: { bounded: route.models } })
  const p = createProvider({
    adapters: [adapter],
    routes: { primary: { route: 'bounded', model: 'm' } },
    contract: new NullContractStore(),
    secrets: () => 'fixture-key',
    clock: () => 0,
  })
  const gate = vi.fn(async () => true)
  const events = await collect(p.infer(request(), { ...options(), beforeAttempt: gate }))
  expect((p as Provider).supportsAttemptAdmission).toBe(true)
  expect(adapter.calls).toHaveLength(0)
  expect(gate).not.toHaveBeenCalled()
  expect(events).toEqual([quota])
})

it('fails closed on a Pi API whose transport cannot expose every HTTP dispatch', async () => {
  const stream = vi.fn(async function* () {})
  const p = provider({
    manualRoutes: [{ ...route, api: 'google-generative-ai' }],
    streamImpl: stream,
  })
  const gate = vi.fn(async () => true)
  const events = await collect(p.infer(request(), { ...options(), beforeAttempt: gate }))
  expect(stream).not.toHaveBeenCalled()
  expect(gate).not.toHaveBeenCalled()
  expect(events).toEqual([quota])
})

it.each(['image', 'video'] as const)(
  'fails closed for opaque %s generation before vendor calls',
  async (slot) => {
    const image = fakeModel({ id: 'm', route: 'agnes-media-image' })
    const video = fakeModel({ id: 'm', route: 'agnes-media-video' })
    const imagesImpl = vi.fn(async () => ({ images: [] }))
    const submit = vi.fn(async () => ({ jobId: 'fixture' }))
    const poll = vi.fn(async () => ({ status: 'succeeded' as const, url: 'https://fixture.invalid' }))
    const cancel = vi.fn(async () => {})
    const adapter = new MediaAdapter({
      imageRoute: { baseUrl: 'https://media.invalid', models: [image] },
      videoRoute: { baseUrl: 'https://media.invalid', models: [video] },
      imagesImpl,
      videoClient: { submit, poll, cancel },
    })
    const gate = vi.fn(async () => true)
    const selected = slot === 'image' ? image : video
    const events = await collect(
      adapter.stream(selected.route, fakeRequest({ slot, route: selected.route, model: 'm' }), {
        ...options(),
        beforeAttempt: gate,
        sessionKey: 'fixture',
        timeoutMs: { firstToken: 1000, total: 5000 },
      }),
    )
    expect(imagesImpl).not.toHaveBeenCalled()
    expect(submit).not.toHaveBeenCalled()
    expect(poll).not.toHaveBeenCalled()
    expect(cancel).not.toHaveBeenCalled()
    expect(gate).not.toHaveBeenCalled()
    expect(events).toEqual([quota])
  },
)
