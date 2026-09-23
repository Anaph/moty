"""Tokenizer parity corpus for tests/build/tok_oracle.

  tok_cases.py <tokenizer.json> [text files...] > cases.tsv

Emits lines TEXT\\tID,ID,... (TEXT with \\n \\t \\r \\\\ escaped) encoded by
HF `tokenizers` without special tokens: a fixed set of edge cases (digit
runs next to whitespace, contractions, CJK, Cyrillic, emoji, code) plus
random slices of the given text files."""
import sys, random
from tokenizers import Tokenizer
tk = Tokenizer.from_file(sys.argv[1])
random.seed(7)
texts = [
 "x  5", "x   55555 y", "a1b22c333d4444", "  12  345   6789 ", "Price: $1,234,567.89!", "3rd place, 2nd try, 10000th run",
 "tab\there\tand  double  spaces", "line1\nline2\r\nline3\n\n\n", "   leading and trailing   ", "emoji 😀🚀 ok", "中文测试，数字123和英文abc",
 "Привет, мир! 2026 год.", "it's they're we've I'm you'll he'd", "IT'S THEY'RE", "hello\u00a0world",
 "def f(x):\n    return x**2  # comment\n", "a\n  1\n\t2", "   3", "x\n 7", "!!!???...", "  \n  \n", "1.5e-3 and 0x1F",
]
for f in sys.argv[2:]:
    s = open(f, encoding="utf-8", errors="replace").read()
    for _ in range(6):
        i = random.randrange(0, max(1, len(s) - 200)); texts.append(s[i:i + random.randint(20, 400)])
def esc(t): return t.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
for t in texts:
    print(esc(t) + "\t" + ",".join(map(str, tk.encode(t, add_special_tokens=False).ids)))
