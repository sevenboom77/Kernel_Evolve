import type { Provider, RequestBody } from '@agnes/protocol'
import type { SessionLogImpl } from '../log/session-log.js'
import { CoreError, type Event } from '../types.js'
import type { SessionImpl } from './session.js'

const REGISTER = 'x/core/provider-request-budget'
const ADMITTED = 'x/core/provider-request-admitted'
const LANE = 'main'
type Budget = { sessionKey: string; cap: number; used: number }
type SharedBudget = {
  queue: Promise<void>
  initialized: boolean
  budget: Budget | undefined
  configured: number | undefined
}
const gates = new WeakMap<SessionLogImpl, SharedBudget>()

function positiveInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) > 0
}

/** A reservation is never refunded: a crash cannot prove that its dispatch did not happen. */
export class ProviderRequestBudget {
  private readonly shared: SharedBudget

  constructor(
    private readonly session: SessionImpl,
    configured?: number,
  ) {
    if (configured !== undefined && !positiveInteger(configured))
      throw new CoreError('E_ENVELOPE', 'session.max_provider_requests must be a positive safe integer')
    const previous = gates.get(session.d.log)
    if (previous?.initialized && !previous.budget && configured !== undefined)
      throw new CoreError(
        'E_ENVELOPE',
        'enabling a provider request cap requires reopening the session writer',
      )
    this.shared = previous ?? { queue: Promise.resolve(), initialized: false, budget: undefined, configured }
    if (
      previous &&
      configured !== undefined &&
      (previous.configured === undefined || configured < previous.configured)
    ) {
      previous.configured = configured
      previous.initialized = false
    }
    gates.set(session.d.log, this.shared)
  }

  private serialized<T>(work: () => Promise<T>): Promise<T> {
    const result = this.shared.queue.then(work)
    this.shared.queue = result.then(
      () => undefined,
      () => undefined,
    )
    return result
  }

  initialize(): Promise<void> {
    return this.serialized(() => this.initializeInsideQueue())
  }

  private async initializeInsideQueue(): Promise<void> {
    if (this.shared.initialized) return
    const { session } = this
    // These are ignorable ledger rows, not a new materialized register: openTracked rebuilds only
    // the known protocol registers. The verified ledger itself is the durable source of truth.
    const last = await Promise.all(
      [REGISTER, ADMITTED].map(async (type) => {
        let toSeq = session.lastSeq
        while (toSeq > 0) {
          const rows = await session.scan({ type, toSeq, order: 'desc', limit: 100 })
          const trusted = rows.find((event) => event.origin === 'system' && event.trust === 'trusted')
          if (trusted) return trusted
          if (!rows.length) return undefined
          toSeq = (rows.at(-1)?.seq ?? 1) - 1
        }
        return undefined
      }),
    )
    if (last.some(Boolean) || this.shared.configured !== undefined) {
      let restored: Budget | undefined
      let legacy = false
      let fromSeq = 1
      const toSeq = session.lastSeq
      while (fromSeq <= toSeq) {
        const rows = await session.scan({ fromSeq, toSeq, limit: 100 })
        if (!rows.length) break
        for (const event of rows) {
          if (event.origin !== 'system' || event.trust !== 'trusted') continue
          if ([REGISTER, ADMITTED].includes(event.type)) {
            const next = this.read(event)
            if (
              (!restored && event.type === ADMITTED) ||
              (restored &&
                (next.cap > restored.cap || next.used !== restored.used + (event.type === ADMITTED ? 1 : 0)))
            )
              throw new CoreError('E_ENVELOPE', 'provider request budget history changed allowance')
            restored = next
          } else if (!restored) {
            const kind = (event.data as { kind?: string } | null)?.kind
            if (
              event.type === 'request/sent' ||
              event.type === 'cost/ledger' ||
              (event.type === 'effect/intent' && (kind === 'inference' || kind === 'compaction'))
            )
              legacy = true
          }
        }
        fromSeq = (rows.at(-1)?.seq ?? toSeq) + 1
      }
      if (restored) {
        this.shared.budget = {
          ...restored,
          cap: Math.min(restored.cap, this.shared.configured ?? restored.cap),
        }
        if (this.shared.budget.cap !== restored.cap) await this.write(REGISTER, this.shared.budget)
      } else if (this.shared.configured !== undefined) {
        // Unknown past physical wire work consumes the whole new allowance, rather than resetting.
        this.shared.budget = {
          sessionKey: session.key,
          cap: this.shared.configured,
          used: legacy ? this.shared.configured : 0,
        }
        await this.write(REGISTER, this.shared.budget)
      }
    }
    this.shared.initialized = true
  }

  private read(event: Event): Budget {
    const data = event.data as Budget | null
    if (
      !data ||
      event.lane !== LANE ||
      data.sessionKey !== this.session.key ||
      !positiveInteger(data.cap) ||
      !Number.isSafeInteger(data.used) ||
      data.used < 0
    )
      throw new CoreError('E_ENVELOPE', 'invalid persisted provider request budget')
    return { sessionKey: data.sessionKey, cap: data.cap, used: data.used }
  }

  private async write(type: string, data: Budget & Record<string, unknown>) {
    await this.session.d.log.append([
      {
        ...this.session.ev(type, data, { ignorable: true }),
        lane: LANE,
      },
    ])
  }

  get enabled(): boolean {
    return this.shared.budget !== undefined
  }

  /** A copied informational view; reading it neither reserves nor refunds an attempt. */
  snapshot(): { cap: number; used: number; remaining: number } | undefined {
    const budget = this.shared.budget
    return budget
      ? { cap: budget.cap, used: budget.used, remaining: Math.max(0, budget.cap - budget.used) }
      : undefined
  }

  available(provider: Provider): boolean {
    return (
      !this.shared.budget ||
      (provider.supportsAttemptAdmission === true && this.shared.budget.used < this.shared.budget.cap)
    )
  }

  private admit(request: RequestBody, signal: AbortSignal, effectId?: string): Promise<boolean> {
    return this.serialized(async () => {
      await this.initializeInsideQueue()
      if (signal.aborted || this.session.closingOrClosed) return false
      const current = this.shared.budget
      if (!current) return true
      if (current.used >= current.cap) return false
      const next = { ...current, used: current.used + 1 }
      await this.write(ADMITTED, {
        ...next,
        requestLane: this.session.lane,
        slot: request.slot,
        model: request.model,
        route: request.route,
        derivedHash: request.derivedHash,
        requestKind: request.kind,
        ...(effectId ? { effectId } : {}),
      })
      this.shared.budget = next
      return !signal.aborted && !this.session.closingOrClosed
    })
  }

  wrap(provider: Provider): Provider {
    const budget = this
    const count = provider.count?.bind(provider)
    return {
      get supportsAttemptAdmission() {
        return provider.supportsAttemptAdmission
      },
      models: () => provider.models(),
      ...(count ? { count } : {}),
      async *infer(request, options) {
        await budget.initialize()
        if (!budget.enabled) {
          yield* provider.infer(request, options)
          return
        }
        if (provider.supportsAttemptAdmission !== true) {
          yield {
            type: 'error',
            reason: 'error',
            code: 'QUOTA',
            message: 'provider attempt admission unavailable',
            retryable: false,
          }
          return
        }
        const phase = budget.session.op()?.phase
        const effectId =
          phase?.kind === 'inference' && phase.gen.status === 'effect_pending'
            ? phase.gen.effectId
            : phase?.kind === 'compaction'
              ? phase.effectIds?.[0]
              : undefined
        yield* provider.infer(request, {
          ...options,
          beforeAttempt: async ({ signal }) => {
            if (!(await budget.admit(request, signal, effectId))) return false
            return options.beforeAttempt ? options.beforeAttempt({ signal }) : true
          },
        })
      },
    }
  }
}
