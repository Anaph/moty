# HF transformers reference tooling (validation of moty engines)

Needs `torch` + `transformers` (+ `tokenizers`); x86 is fine. The engines
consume the outputs through `REF=`, `REF_LOGITS=`, `PPL=` / `PPL_OUT=`
(see the README env table).

| script | output | consumer |
|---|---|---|
| `hf_ref.py <snap> <out.json> [ngen]` | per prompt: prompt_ids, greedy full_ids, last-position logits; plus `<out>.caseN.json` | `REF=<out>.caseN.json`; `cmp_logits.py` |
| `hf_ppl.py <snap> <out.json> [text] [max_tokens]` | ids of the text (default `ppl_text.txt`), per-position argmax, perplexity (f32) | `PPL=<out.json>`, `cmp_ppl.py` |
| `cmp_logits.py <ref.json> <case> <REF_LOGITS file>` | max \|dlogit\|, top-1 / top-10 agreement | |
| `cmp_ppl.py <ppl.json> <PPL_OUT file>` | top-1 agreement vs f32 HF | |
| `tok_cases.py <tokenizer.json> [files...]` | `TEXT\tIDS` corpus (edge cases + slices of the files) | `tests/build/tok_oracle` |

Texts: `ppl_text.txt` (~290 tokens: prose, a story, Python, Russian) for
quick checks; `ppl_text_long.txt` (the same followed by an older copy of
`docs/architecture.md`, >2048 tokens) for quality numbers — on short texts
the ranking of int4 scale rules flipped with noise, 2048 tokens are stable.

Example (LFM2.5-350M, f32 exactness, then int4 quality):

    python hf_ref.py $M lfm_ref.json 32
    SNAP=$M REF=lfm_ref.case0.json REF_LOGITS=/tmp/l.bin QBITS=0 IDOT=0 ./lfm2
    python cmp_logits.py lfm_ref.json 0 /tmp/l.bin
    python hf_ppl.py $M lfm_ppl.json ppl_text_long.txt 2048
    SNAP=$M PPL=lfm_ppl.json PPL_OUT=/tmp/am.txt QBITS=4 ./lfm2
    python cmp_ppl.py lfm_ppl.json /tmp/am.txt
