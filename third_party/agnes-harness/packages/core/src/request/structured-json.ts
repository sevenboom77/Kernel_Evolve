import { canonicalJson } from './hash.js'

/** Compact only a complete JSON view of the tool's structured result, preserving its lexemes. */
export function compactStructuredJson(text: string, structured: unknown): string {
  if (structured === null || typeof structured !== 'object') return text
  try {
    const parsed: unknown = JSON.parse(text)
    if (parsed === null || typeof parsed !== 'object' || canonicalJson(parsed) !== canonicalJson(structured))
      return text
  } catch {
    return text
  }

  // Re-serializing would change large numbers, negative zero, duplicate keys and escapes.
  // The parser above establishes valid JSON; remove only JSON whitespace outside strings.
  let compact = ''
  let quoted = false
  let escaped = false
  for (const char of text) {
    if (quoted) {
      compact += char
      if (escaped) escaped = false
      else if (char === '\\') escaped = true
      else if (char === '"') quoted = false
    } else if (char === '"') {
      quoted = true
      compact += char
    } else if (char !== ' ' && char !== '\t' && char !== '\r' && char !== '\n') {
      compact += char
    }
  }
  return compact
}
