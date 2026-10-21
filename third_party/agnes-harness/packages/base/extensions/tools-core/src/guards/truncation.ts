// The guard against the most expensive failure a writing tool has: a model that meant to rewrite a
// file and emitted only part of it. The two signals are cheap and independent - the new content is
// a small fraction of what it replaces, or it closes fewer brackets and quotes than the original
// did - and both are refusals rather than repairs, because the tool cannot know what was dropped.

const PAIRS: Array<[string, string]> = [
  ['{', '}'],
  ['[', ']'],
  ['(', ')'],
]

// Below this fraction of the original, a rewrite is treated as a suspected truncation. Sitting
// exactly on it is accepted: shrinking a file by well over half is a normal edit, and the threshold
// only has to catch the case where most of the file is simply gone.
const MIN_RATIO = 0.4

function count(s: string, ch: string): number {
  return s.split(ch).length - 1
}

// The fallback for unknown languages retains the original character-count heuristic. Counts are
// compared against the original, so a template fragment is not held to a balance it never had.
function unbalanced(s: string): string[] {
  const bad: string[] = []
  for (const [open, close] of PAIRS) if (count(s, open) !== count(s, close)) bad.push(open)
  if (count(s, '"') % 2 !== 0) bad.push('"')
  return bad
}

// Python brackets in strings and comments are prose, not code. This is a lexical completeness
// check, not a syntax parser: it recognizes quoting without interpreting f-string expressions.
// Keep unclosed strings as an imbalance rather than hiding a genuinely partial model response.
function pythonUnbalanced(s: string): string[] {
  let code = ''
  const strings: string[] = []
  let i = 0
  while (i < s.length) {
    const ch = s.charAt(i)
    if (ch === '#') {
      while (i < s.length && s[i] !== '\n' && s[i] !== '\r') i++
    } else if (ch === '"' || ch === "'") {
      const delimiter = s.startsWith(ch.repeat(3), i) ? ch.repeat(3) : ch
      i += delimiter.length
      let closed = false
      while (i < s.length) {
        if (s[i] === '\\') {
          // Escaped quotes do not terminate a string. Python also permits escaped newlines.
          i += s[i + 1] === '\r' && s[i + 2] === '\n' ? 3 : 2
        } else if (s.startsWith(delimiter, i)) {
          i += delimiter.length
          closed = true
          break
        } else if (delimiter.length === 1 && (s[i] === '\n' || s[i] === '\r')) {
          break
        } else {
          i++
        }
      }
      if (!closed) strings.push(ch)
    } else {
      code += ch
      i++
    }
  }
  return [...unbalanced(code), ...strings]
}

export function looksTruncated(
  oldText: string,
  newText: string,
  path = '',
): { truncated: false } | { truncated: true; reason: string } {
  // Nothing to compare against: writing into a file that is empty or absent cannot lose anything.
  if (oldText.length === 0) return { truncated: false }
  // Emptying a file is not exempt. It is the shortest kind of short, and an exemption for it let
  // an empty string overwrite any file in the workspace with the guard looking on. Someone who
  // genuinely wants a file emptied says so twice, or removes the content with `edit`; the other
  // direction costs the whole file with no way to tell it happened.
  if (newText.length < MIN_RATIO * oldText.length)
    return {
      truncated: true,
      reason: `new content is ${Math.round((100 * newText.length) / oldText.length)}% of the original`,
    }
  const check = /\.pyi?$/i.test(path) ? pythonUnbalanced : unbalanced
  const before = new Set(check(oldText))
  for (const ch of check(newText)) if (!before.has(ch)) return { truncated: true, reason: `unbalanced ${ch}` }
  return { truncated: false }
}
