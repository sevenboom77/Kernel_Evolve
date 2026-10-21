import type { WireEvent } from './adapter.js'

/** Callback and transport details must never become the public budget failure message. */
export function attemptDenied(): Extract<WireEvent, { type: 'error' }> {
  return {
    type: 'error',
    reason: 'error',
    code: 'QUOTA',
    message: 'provider request limit reached',
    retryable: false,
  }
}
