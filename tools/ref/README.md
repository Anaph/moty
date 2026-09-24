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
| `hf_qsim.py <snap> <schemes> [text]` | HF fake-quant of every Linear with int4 group-32 scale rules (`amax7`, `q40`, `noclip`, `@64` = group 64): PPL, KL vs f32, top-1 | choosing the Q4R4 scale rule |
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

## Quantization study and calibrated containers

| script | what |
|---|---|
| `hf_sens.py <snap> <text>` | per-group sensitivity: one tensor group (a Linear-name regex, `head`, `head8`, `embed8`) fake-quantized to int4 at a time, the rest f32: PPL, KL(f32‖q), top-1 |
| `hf_qstudy.py <snap> <eval> <calib> <config>...` | whole-model fake-quant: schemes (`sym4g32` = Q4R4, g16/64/128, `asym4g32`, `sym4g32i8`, `sym5g32`, `sym8g32`, `q8row`), methods (`:imx`, `:awq`, `:gptq`), mixes (`+Q=<regex>`: int8 group-32 = Q8R4, `+8=`, `+5=`), `@a8`: moty's int8 group-32 activations |
| `pack_r4.py <snap> <out> --method rtn\|gptq --calib calib_text.txt --q8 <globs>` | writes a moty container (the SAVE_PACKED layout): Q4R4, and Q8R4 for tensors matching the globs (the same as moty's `Q8_TENSORS`); `rtn` is byte-identical to moty's own packer, `gptq` = Hessian error-compensated codes |

`calib_text.txt` (calibration only; disjoint from `ppl_text_long.txt`):
docs/online-learning.md, the start of docs/gemma-plan.md, c/nn/ffn.c and
the start of c/nn/conv.c of this repository.

Recommended LFM2.5-350M container for Cortex-A53 (docs/performance.md §5.8):

    python pack_r4.py $SNAP out --method gptq --calib calib_text.txt \
        --q8 "model.layers.0.*,model.layers.1.*"
