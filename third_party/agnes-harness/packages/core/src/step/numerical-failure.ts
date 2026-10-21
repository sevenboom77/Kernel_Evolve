import { canonicalJson, sha256Hex } from '../request/hash.js'

export const numericRepairFocus = {
  large_scale_mismatch:
    'Check sum versus mean, reduction axes and normalization denominators against the reference before tuning blocks or warps.',
  pointwise_tolerance_mismatch:
    'Use the unchanged atol + rtol * abs(reference) at each failing coordinate. Global max_tol is not that coordinate tolerance. Check reference operation order and intermediate casts or rounding.',
  output_contract_mismatch:
    'Derive output shape from runtime inputs and preserve the reference dtype, layout and return arity before speed tuning.',
} as const
export type NumericalFailureClass = keyof typeof numericRepairFocus

export function isNumericalFailureClass(value: unknown): value is NumericalFailureClass {
  return typeof value === 'string' && Object.hasOwn(numericRepairFocus, value)
}

function record(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined
}

// Parse only the bounded standard diagnostic grammar, never Python expressions or free text.
const number = '-?(?:0|[1-9]\\d*)(?:\\.\\d+)?(?:e[+-]?\\d+)?'
const numeric = new RegExp(
  "^output\\[\\d+\\] mismatch: \\{'ok': False, 'n_bad': (\\d+), 'n_total': (\\d+), " +
    `'max_abs_err': (${number}), 'max_tol': (${number}), 'first_bad_flat': (\\d+), ` +
    `'expected': (${number}), 'actual': (${number})` +
    "(?:, 'first_bad_index': \\[(?:\\d+(?:, \\d+)*)?\\])?\\}$",
)
const contract =
  /^output\[\d+\] mismatch: \{'ok': False, 'reason': '(?:shape \((?:\d+(?:, \d+)*,?)?\) != expected \((?:\d+(?:, \d+)*,?)?\)|dtype torch\.[a-z0-9_]+ != expected torch\.[a-z0-9_]+)'\}$|^arity \d+ != \d+$/

function parseNumericProblem(problem: unknown) {
  if (typeof problem !== 'string' || problem.length > 2048) return undefined
  const match = numeric.exec(problem)
  if (!match) return undefined
  const bad = Number(match[1])
  const total = Number(match[2])
  const maxError = Number(match[3])
  const maxTolerance = Number(match[4])
  const flat = Number(match[5])
  const expected = Number(match[6])
  const actual = Number(match[7])
  if (
    ![bad, total, flat].every(Number.isSafeInteger) ||
    ![maxError, maxTolerance, expected, actual].every(Number.isFinite) ||
    bad < 1 ||
    bad > total ||
    total < 1 ||
    flat >= total ||
    maxError <= 0 ||
    maxTolerance < 0 ||
    Math.abs(expected - actual) > maxError * 1.000001
  )
    return undefined
  return [bad, total, maxError, maxTolerance, flat, expected, actual] as const
}

function classifyProblem(problem: unknown): NumericalFailureClass | undefined {
  if (typeof problem !== 'string' || problem.length > 2048) return undefined
  if (contract.test(problem)) return 'output_contract_mismatch'
  const parsed = parseNumericProblem(problem)
  if (!parsed) return undefined
  const [bad, total, maxError, maxTolerance, , expected, actual] = parsed
  const ratio = actual / expected
  if (bad / total >= 0.8 && Number.isFinite(ratio) && ratio > 0 && (ratio <= 0.5 || ratio >= 2))
    return 'large_scale_mismatch'
  if (bad <= Math.max(8, Math.ceil(total * 0.001)) && maxError < maxTolerance)
    return 'pointwise_tolerance_mismatch'
  return undefined
}

function failedCorrectness(value: unknown): Record<string, unknown> | undefined {
  const correctness = record(value)
  if (
    correctness?.ok !== false ||
    typeof correctness.passed !== 'number' ||
    typeof correctness.total !== 'number' ||
    !Number.isSafeInteger(correctness.passed) ||
    !Number.isSafeInteger(correctness.total) ||
    correctness.passed < 0 ||
    correctness.passed >= correctness.total ||
    !Array.isArray(correctness.first_failures) ||
    correctness.first_failures.length > 5
  )
    return undefined
  return correctness
}

/** Categories are search hypotheses, not inferred mathematical causes or acceptance changes. */
export function numericalFailureClass(value: unknown): NumericalFailureClass | undefined {
  const correctness = failedCorrectness(value)
  if (!correctness) return undefined
  const classes = new Map<NumericalFailureClass, Set<string>>()
  for (const failure of correctness.first_failures as unknown[]) {
    const row = record(failure)
    if (
      typeof row?.case_id !== 'string' ||
      !/^[A-Za-z0-9_.:-]{1,128}$/.test(row.case_id) ||
      !Array.isArray(row.problems) ||
      row.problems.length > 4
    )
      continue
    for (const problem of row.problems) {
      const category = classifyProblem(problem)
      if (!category) continue
      const cases = classes.get(category) ?? new Set<string>()
      cases.add(row.case_id)
      classes.set(category, cases)
    }
  }
  if (classes.size !== 1) return undefined
  const entry = classes.entries().next().value
  if (!entry) return undefined
  const [category, cases] = entry
  return cases.size >= (category === 'output_contract_mismatch' ? 1 : 2) ? category : undefined
}

/** Exact bounded diagnostic equality is evidence of repetition, never evidence of its cause. */
export function numericalFailureFingerprint(value: unknown): string | undefined {
  const correctness = failedCorrectness(value)
  if (!correctness || !(correctness.first_failures as unknown[]).length) return undefined
  const rows: [string, string[]][] = []
  for (const failure of correctness.first_failures as unknown[]) {
    const row = record(failure)
    if (
      typeof row?.case_id !== 'string' ||
      !/^[A-Za-z0-9_.:-]{1,128}$/.test(row.case_id) ||
      !Array.isArray(row.problems) ||
      row.problems.length < 1 ||
      row.problems.length > 4 ||
      row.problems.some(
        (problem) =>
          typeof problem !== 'string' ||
          problem.length > 2048 ||
          !(contract.test(problem) || parseNumericProblem(problem)),
      )
    )
      return undefined
    rows.push([row.case_id, [...row.problems].sort()])
  }
  if (new Set(rows.map(([caseId]) => caseId)).size !== rows.length) return undefined
  return sha256Hex(
    canonicalJson([correctness.passed, correctness.total, rows.sort(([a], [b]) => (a < b ? -1 : 1))]),
  )
}
