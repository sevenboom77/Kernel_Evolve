import { describe, expect, it } from 'vitest'
import { looksTruncated } from '../src/guards/truncation.js'

describe('looksTruncated', () => {
  it('accepts a comparable rewrite', () => {
    expect(looksTruncated('a'.repeat(100), 'b'.repeat(90))).toEqual({ truncated: false })
  })

  it('accepts a rewrite sitting exactly on the ratio', () => {
    // The boundary is part of the contract: 40% of the original is still a rewrite, and only
    // less than that is a suspected truncation.
    expect(looksTruncated('a'.repeat(100), 'b'.repeat(40))).toEqual({ truncated: false })
  })

  it('rejects < 40% of original and says how much is left', () => {
    expect(looksTruncated('a'.repeat(100), 'b'.repeat(30))).toMatchObject({
      truncated: true,
      reason: expect.stringContaining('30%'),
    })
  })

  it('rejects newly unbalanced brackets', () => {
    expect(looksTruncated('f(){ return 1 }', 'f(){ return 1')).toMatchObject({
      truncated: true,
      reason: 'unbalanced {',
    })
  })

  it.each([
    ['docstring interval', 'def f():\n    """Offsets lie in [0, C)."""\n    return (1)\n'],
    ['triple single quotes', "def f():\n    '''Offsets lie in [0, C).'''\n    return (1)\n"],
    ['line comment', 'def f():\n    # [0, C), "quoted"\n    return (1)\n'],
    ['escaped double quote', 'value = "escaped \\" and [0, C)"\n'],
    ['escaped single quote', "value = 'it\\'s [0, C)'\n"],
    ['quoted comment marker', 'value = "#"\nresult = (1)\n'],
    ['multiline docstring', 'def f():\n    """A "quote"\n    [0, C)\n    """\n    return (1)\n'],
    ['escaped triple quote', 'value = """escaped \\""" and [0, C)"""\n'],
  ])('accepts Python %s without counting its literal delimiters', (_name, source) => {
    expect(looksTruncated('value = 1\n', source, 'kernel.py')).toEqual({ truncated: false })
  })

  it.each([
    ['missing code parenthesis', 'value = "[0, C)"\nresult = (1\n'],
    ['division before missing bracket', 'value = 10 // 2\nresult = [1\n'],
    ['missing closing double quote', 'value = "unterminated (\n'],
    ['missing closing single quote', "value = 'unterminated (\n"],
    ['missing closing triple quote', 'value = """unterminated [0, C)\n'],
    ['escaped closing quote', 'value = "unterminated \\"'],
    ['unescaped newline in a short string', 'value = "line\nnext"\n'],
  ])('still rejects Python %s', (_name, source) => {
    expect(looksTruncated('value = 1\n', source, 'kernel.py')).toMatchObject({ truncated: true })
  })

  it('keeps non-Python delimiter checks and the existing-fragment exemption', () => {
    expect(looksTruncated('value = 1\n', 'value = "[0, C)"\n', 'notes.txt')).toMatchObject({
      truncated: true,
    })
    expect(looksTruncated('value = "open', 'other = "open', 'kernel.pyi')).toEqual({ truncated: false })
    expect(looksTruncated('x'.repeat(100), '# complete comment\n', 'kernel.py')).toMatchObject({
      truncated: true,
      reason: expect.stringContaining('%'),
    })
  })

  it('lets an already unbalanced original stay unbalanced', () => {
    expect(looksTruncated('x = "(', 'y = "(')).toEqual({ truncated: false })
  })

  it('accepts writes into empty files', () => {
    expect(looksTruncated('', 'anything')).toEqual({ truncated: false })
    expect(looksTruncated('', '')).toEqual({ truncated: false })
  })

  it('rejects emptying a non-empty file', () => {
    // Truncating to nothing is the case this guard exists for, and an exemption for empty new
    // content made `write` with an empty string succeed over any file in the workspace. Empty is
    // the shortest kind of short.
    expect(looksTruncated('x'.repeat(100_000), '')).toMatchObject({
      truncated: true,
      reason: expect.stringContaining('0%'),
    })
    expect(looksTruncated('hello', '')).toMatchObject({ truncated: true })
  })
})
