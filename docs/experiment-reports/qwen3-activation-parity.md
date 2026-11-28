# Does our engine compute the same activations as Hugging Face? A day chasing Qwen3-1.7B's outliers

*28 November 2026*

When we harvest activations with our engine, we need them to be the activations the model really
produces, not something our fused kernels made up. For Llama-3.2-1B, the engine has always matched
Hugging Face (HF) closely. For Qwen3-1.7B, earlier checks kept flagging comparisons that failed, and
a few tokens were far off. This report covers what we did on 28 November to find out why, and where
we landed.

## Setup

| | |
|---|---|
| Model | Qwen3-1.7B, checkpoint revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. 28 blocks, hidden width 2,048, 16 query heads and 8 key/value heads of size 128, with Q/K normalization |
| GPU | One NVIDIA Tesla T4 on Modal, one fresh container per job, one request at a time |
| Software | PyTorch 2.8.0 (CUDA 12.8), Transformers 4.57.3, FlashInfer 0.5.3 |
| Our engine | FP16. FlashInfer FA2 attention, fused residual add + RMSNorm, merged Q/K/V and gate/up projections, fused SiLU-and-multiply |
| HF reference | FP16 with eager attention. For the FP32 check, the same model in FP32 with eager attention and TF32 switched off |
| Prompts | 100 first-turn user prompts from WildChat-4.8M (revision `c827c6df`), English or Chinese, toxic and redacted conversations removed, duplicates removed. Seed 42, 25 prompts from each length range: 1–256, 257–1,024, 1,025–2,048 and 2,049–3,072 tokens. Lengths ranged from 14 to 3,055 tokens, 115,303 tokens in total |
| Prefill | The whole prompt in one forward pass on both sides. Both received the same saved token IDs. The engine samples one token only to trigger prefill; decode was not compared |
| Capture points | 57 per token: the attention output after the output projection, and the residual stream after each completed block, for all 28 blocks, plus the final normalized hidden state that goes into the LM head |

**Metrics.** From 28 November on, every comparison is per token and per capture point, computed in
FP64 on CPU. (The older 1 November checks below also used whole-tensor measures.)

- **Raw L2:** the length of the difference between our vector and the reference vector. It is in
  activation units, so it grows with how large the activations are.
- **Relative error:** raw L2 divided by the length of the reference vector at that token. A relative
  error of 1% means the difference is 1% the size of the activation itself.
- **Reference:** HF FP16 at first. In the last two experiments, HF FP32 with TF32 off. FP32 is our
  higher-precision stand-in for "the true model". It is still rounded, just far less than FP16,
  which keeps only about three significant digits.

## What made us look

Earlier runs on 1 November compared the engine (256-token chunked prefill) against HF (whole
prompt) on 50 prompts per model. They used three pass/fail checks: relative L2 at most 5%,
whole-tensor cosine at least 0.999, and the worst single token's cosine at least 0.98.

| Model | Failed comparisons | Prompts with a failure | Largest whole-tensor relative L2 |
|---|---:|---:|---:|
| Llama-3.2-1B | 0 of 1,716 | 0 of 50 | 0.38% |
| Llama-3.2-3B | 13 of 2,964 | 4 of 50 | 0.82% |
| Qwen3-1.7B | 125 of 2,964 | 12 of 50 | 2.49% |
| Qwen3-4B | 442 of 3,796 | 18 of 50 | 20.69% |

All 125 Qwen3-1.7B failures came from the worst-token cosine check; none failed the whole-tensor
checks. The tensor as a whole was close, but a single token was far off. On 24 November we tested
two easy explanations:

- **Weights.** The engine loads exactly HF's weights. All 311 tensors were bit-identical, and so were
  the norm settings and head counts.
- **Fused residual add + RMSNorm.** On the prompts that had failed, splitting it into HF-style
  separate steps moved the failures only from 580 to 555 across the three affected models. That
  did not resolve the problem. It also did not rule normalization out as one contributor, since it
  changed only the fused calls.

The 1 November runs also processed the prompt differently on each side: chunks on the engine,
whole prompt on HF. So the first thing we did on the 28th was to remove that difference.

## What we did on 28 November

| # | Experiment | What it asked | Where it ran |
|---|---|---|---|
| 1 | 100-prompt whole-prefill run | How far apart are engine and HF FP16 when both process the whole prompt at once? | T4 |
| 2 | Audit of the comparison method | Are we capturing the same thing on both sides, and comparing it correctly? | Code reading |
| 3 | Comparison against HF's source code | Does our engine do the same math as HF's Qwen3? | Code reading |
| 4 | Outlier patterns | What do the outliers have in common? What is the average error with and without them? | Local, saved numbers only |
| 5 | Trace of the worst token | Where does the worst token first differ, and where does the difference grow? | T4 |
| 6 | Operation swaps | Which engine operation, swapped for HF's version, shrinks the worst token's error the most? | T4 |
| 7 | Relative error | Is the worst token's error large relative to the size of the activation? | Modal CPU |
| 8 | FP32 check, one prompt | Measured against FP32, is our engine or HF FP16 closer to the true model? | T4 |
| 9 | FP32 check, six prompts | Does that answer hold on other outlier-heavy prompts? | T4 |

## What we found

### 1. With matched prefill, the typical token is close, but a few tokens are far off

All 100 prompts finished, and every saved record passed its integrity checks: 5,700 prompt and
capture-point comparisons, and 6,572,271 per-token comparisons. Raw per-token L2 against HF FP16:

| Capture point | Median | 95th percentile | 99th percentile | Maximum |
|---|---:|---:|---:|---:|
| Attention output | 0.170 | 1.34 | 2.32 | 127.8 |
| Residual after each block | 0.415 | 5.19 | 9.70 | 799.4 |
| Final normalized state | 0.199 | 0.54 | 1.26 | 35.2 |

The maximum sits hundreds of times above the median. The largest value of all was at the last
block's residual, at token 623 of dataset row 2802, a 1,431-token prompt containing Ruby code. That
token's final normalized state was also the worst: L2 35.2.

### 2. The comparison method is sound

On the engine, the attention output is captured right after the output projection. The block
residual for block *i* is captured after the next block's fused add, which makes it the complete
output of block *i*. The last block's residual is captured at the final norm. The final normalized
state is captured for every prompt token, before the engine keeps only the last token. Captures are
copied before later operations overwrite those tensors in place. HF's hooks sit at the same three
places: the output projection, the block output and the final norm. The difference and its length
are computed in FP64 after both tensors are copied to CPU.

A misaligned capture would most likely show large errors from block 0 onward. Instead, the error
starts tiny (mean residual L2 0.017 at block 0) and grows through the blocks, the pattern you
would expect from accumulated rounding differences. The trace in finding 5 adds direct checks on
one prompt. Those checks support correct alignment, but none of this proves there are no bugs at
all.

### 3. The engine does the same math as HF; it just rounds at different places

Line by line against HF's Qwen3 code, both follow the same block structure and the same formulas.
They differ in how the numbers are carried through: where results are rounded to FP16, which steps
are fused into one kernel, and how matrix multiplies are shaped.

| Operation | HF FP16 | Our engine |
|---|---|---|
| RMSNorm | Normalizes in FP32, rounds to FP16, then multiplies by the weight | Multiplies by the weight in FP32, then rounds once |
| Rotary embedding | Uses cos/sin tables rounded to FP16 | Uses cos/sin tables kept in FP32, applied inside one kernel |
| Attention | Eager: the QKᵀ scores are stored in FP16 before scaling and softmax | FlashAttention-2: scores stay in FP32 inside the kernel |
| SiLU × up | SiLU rounded to FP16, then the multiply rounded again | One fused kernel, rounded once |
| Q/K/V and gate/up | Separate matrix multiplies | One merged matrix multiply each, which can change the summation order but not the formula |

In short, several of the engine's fused kernels skip an intermediate FP16 rounding that HF FP16
performs. So the real question is not "why don't they match?" but "which of the two is closer to
the true model?" Experiments 8 and 9 answer it.

### 4. The outliers are rare, sit late in long prompts, and fall on repeated or copyable text

Removing the largest 0.1% or 1% of values barely moves the average, so the outliers are rare and do
not drive the mean:

| Capture point | Mean, all tokens | Mean without the top 0.1% | Mean without the top 1% | Median |
|---|---:|---:|---:|---:|
| Attention output | 0.404 | 0.391 | 0.364 | 0.170 |
| Residual after each block | 1.467 | 1.403 | 1.278 | 0.415 |
| Final normalized state | 0.272 | 0.262 | 0.245 | 0.199 |

The error grows through the blocks. Mean raw L2 of the block residual:

| Block | 0 | 4 | 8 | 12 | 16 | 20 | 24 | 27 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mean residual L2 | 0.017 | 0.075 | 0.155 | 0.366 | 0.889 | 2.218 | 4.209 | 6.414 |

Some of this growth may simply be the residual stream itself getting larger in later blocks. This
run did not save activation sizes, so we can't say how much.

We took the 116 tokens in the top 0.1% of final-state L2 and looked at what they have in common:

- **None at the very start of a prompt.** Not one is among the first three tokens, the chat template.
  Half sit beyond 67% of the way through their prompt.
- **Long prompts.** By prompt length, 0 fall in prompts of up to 256 tokens, 5 in 257–1,024, 36 in
  1,025–2,048 and 75 in prompts longer than 2,048 tokens.
- **Concentrated in a few prompts.** They sit in only 34 of the 100 prompts. Row 2555 alone holds 23.
- **Repeated or copyable text.** Digits, brackets and repeated identifiers in code and data are far
  more common among outliers than in the prompts overall. Examples: `lag` (155 times more common,
  from a list like `'lagX_8', 'lag…`), `(` (16 times), `0` (8.7 times) and `1` (4.4 times).
- **They break away in the middle blocks.** For each of the 20 worst tokens, the attention-output
  error first exceeds 1.0 between blocks 8 and 12. After that it stays high, rising and falling
  from block to block.

These patterns describe where the outliers are. They don't explain why these particular tokens
amplify small differences.

### 5. Tracing the worst token: a tiny rounding difference in block 0 that grows block by block

We traced row 2802, token 623 (the token `-`) at 18 points inside every block. Before trusting the
trace, we checked three things:

- The original outlier reproduced exactly.
- Adding the extra hooks changed none of the 57 standard captures.
- The embeddings were identical on both sides.

The first difference appears at block 0's input normalization, with an L2 of 0.00158. Moving HF's
rounding step to after the weight multiply makes token 623 match exactly there. So for this token,
the first difference is the RMSNorm rounding order from finding 3. (Across the whole prompt, a
smaller difference remains after that change: whole-prompt L2 0.00104.) From there, the difference
grows block by block:

| Completed block | 0 | 4 | 8 | 12 | 14 | 16 | 20 | 24 | 27 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Residual L2 at token 623 | 0.008 | 0.13 | 0.46 | 6.2 | 38.2 | 120.4 | 302.3 | 451.1 | 799.4 |

We also replayed each traced engine operation on HF's saved inputs, to see the difference each
operation adds by itself. The residual additions and the output, gate, up and down projections came
out exactly equal to HF, at every block and every token. Normalization, merged Q/K/V, rotary
embedding, attention and fused SiLU each differ by a rounding-sized amount. The largest was
attention, with an L2 of up to 1.29 at some token. For example, block 27's MLP output is 603 L2 away
in the full forward pass, yet its down projection matches HF exactly when given HF's input. So the
large error arrives mostly from earlier blocks, rather than being created by one operation. The
replay used saved tensors, not the exact serving memory layout, so this is strong evidence, not
a complete proof.

### 6. Every operation swap changes the outlier; none removes it

We swapped one family of engine operations at a time, across all blocks, for HF's FP16 version, and
measured token 623's final-state error against HF FP16:

| Swapped for HF's version | Final-state L2 at token 623 | Relative error |
|---|---:|---:|
| Nothing swapped (baseline) | 35.24 | 34.9% |
| Normalization | 15.44 | 15.3% |
| Separate Q/K/V projections | 24.54 | 24.3% |
| Rotary embedding | 23.73 | 23.5% |
| Eager attention | 14.69 | 14.5% |
| Separate SiLU and multiply | 6.57 | 6.5% |
| Everything restored | 35.24 | 34.9% |

Every swap moves this token closer to HF FP16, since it copies more of HF's way of computing.
None removes the gap on its own, and the effects cannot be added up. Being closer to HF FP16 does
not mean being closer to the true model; findings 8 and 9 test that. Restoring the engine reproduced
the baseline exactly, at all 57 points.

### 7. Relative to its size, the worst token really is far off

Token 623's final-state vector has a length of 101.06, so its raw L2 of 35.24 is a relative error
of 34.9%. The median token in the same prompt is off by only 0.27%. At the last block's residual,
the token's error is 49.1% of the vector's length. This is not a big number on a big vector; the
token's state has really moved.

### 8. Against FP32, our engine is closer for most tokens, but this prompt's worst token is not

We ran HF once more in FP32 on row 2802, and compared both FP16 implementations against it.

| Final normalized state, against FP32 | Median | 99th percentile | 99.9th percentile | Maximum |
|---|---:|---:|---:|---:|
| Our engine FP16 | 0.16% | 1.05% | 2.77% | 26.6% |
| HF FP16 | 0.22% | 1.20% | 4.84% | 12.6% |

- **Most tokens: the engine is closer.** It is closer to FP32 for 1,253 of 1,431 tokens, and has the
  lower median error at all 57 capture points.
- **The worst token: HF FP16 is closer.** At token 623, the engine is 26.6% away from FP32 and HF
  FP16 is 12.6% away. In the later blocks, the engine's worst token is worse at 29 of 57 capture
  points:

| Capture point | Engine maximum | HF FP16 maximum |
|---|---:|---:|
| Block 10 attention output | 2.9% | 9.2% |
| Block 15 attention output | 32.5% | 19.6% |
| Block 20 residual | 43.5% | 14.3% |
| Block 27 residual | 30.0% | 16.1% |
| Final normalized state | 26.6% | 12.6% |

- **HF FP16 is also far from FP32 at this token.** Its 12.6% here is its own worst token in the
  prompt, against a median of 0.22%.
- **The two implementations struggle with the same tokens.** Half of each one's 20 worst tokens are
  the same, and their per-token errors move together (correlation 0.83, measured on log scale).

So token 623 is fragile in FP16 whichever implementation runs it. One prompt, though, could not tell
us whether the engine consistently lands further away on such tokens.

### 9. On six prompts, the engine is closer to FP32, including in the tail

We repeated the FP32 check on five more prompts, picked because they have many final-state
outliers: rows 540, 1359, 2197, 2555 and 5632. Each got one engine FP16 pass, one HF FP16 pass and
one HF FP32 pass, with TF32 off. Row 2802's earlier results were reused. The sample deliberately
leans toward outliers, so it is neither random nor a guaranteed worst case.

Each cell below shows engine / HF FP16, as relative error against FP32 at the final normalized state:

| Dataset row | Tokens | Median | 99th percentile | 99.9th percentile | Maximum |
|---|---:|---:|---:|---:|---:|
| 540 | 3,033 | 0.149% / 0.226% | 0.93% / 1.87% | 3.14% / 7.30% | 9.47% / **29.86%** |
| 1359 | 2,746 | 0.132% / 0.189% | 0.66% / 1.71% | 1.82% / 6.98% | 3.79% / 12.90% |
| 2197 | 1,649 | 0.129% / 0.199% | 1.73% / 3.39% | 4.38% / 9.89% | 5.83% / 13.33% |
| 2555 | 2,822 | 0.153% / 0.236% | 1.87% / 3.89% | 6.37% / 13.03% | 10.54% / 18.74% |
| 2802 | 1,431 | 0.155% / 0.220% | 1.05% / 1.20% | 2.77% / 4.84% | **26.61%** / 12.63% |
| 5632 | 2,684 | 0.131% / 0.191% | 0.87% / 1.82% | 3.02% / 5.98% | 9.32% / 18.72% |
| All six | 14,365 | 0.141% / 0.208% | 1.18% / 2.45% | 3.77% / 10.60% | 26.61% / 29.86% |

| Dataset row | Tokens above 2% | Tokens above 5% | Tokens where each side is closer | Same tokens in both worst-20 lists | Points where the engine's maximum is worse |
|---|---:|---:|---:|---:|---:|
| 540 | 10 / 27 | 3 / 5 | 2,882 / 151 | 9 | 0 of 57 |
| 1359 | 2 / 23 | 0 / 4 | 2,589 / 157 | 10 | 0 of 57 |
| 2197 | 13 / 36 | 2 / 9 | 1,601 / 48 | 11 | 0 of 57 |
| 2555 | 25 / 78 | 5 / 22 | 2,728 / 94 | 10 | 0 of 57 |
| 2802 | 5 / 4 | 1 / 2 | 1,253 / 178 | 10 | 29 of 57 |
| 5632 | 4 / 22 | 1 / 5 | 2,574 / 110 | 11 | 0 of 57 |
| All six | 59 / 190 | 12 / 47 | 13,627 / 738 | 9 | 11 of 57 |

The table reads like this:

- **The engine is closer for 95% of tokens.** Pooled over all six prompts, that is 13,627 of
  14,365 tokens. It also has the lower median at every capture point, on every prompt.
- **Its tail is smaller too.** It has about a third as many tokens above 2% (59 against 190) and a
  quarter as many above 5% (12 against 47). Its 99.9th percentile is about a third of HF FP16's.
- **Row 2802 is the only exception.** On the other five prompts, the engine's worst token is closer
  to FP32 than HF FP16's worst token, at all 57 capture points. At the final normalized state, HF
  FP16's single worst token across the six prompts, 29.9% on row 540, is worse than the engine's
  worst, 26.6% on row 2802.
- **Both sides still struggle with the same tokens.** About half of each side's 20 worst tokens are
  shared.

### What it adds up to

We found no capture misalignment and no weight mismatch, and the engine follows the same formulas
as HF. That does not prove there are no bugs at all. But everything we measured fits one picture:
small rounding differences, starting as early as block 0's first normalization, accumulate through
the network, and a few tokens amplify them a great deal. Those tokens tend to be repeated or
copyable text late in long prompts. We don't yet know why those tokens amplify.

HF FP16 suffers from the same thing, and on these prompts it suffers more than our engine does.
Agreement with HF FP16 alone can't tell you how accurate the engine is, because HF FP16 is itself
noisy. Against FP32, on six outlier-heavy prompts:

- the engine has the lower median error at every capture point;
- it is closer to FP32 on 13,627 of 14,365 final-state tokens;
- its worst token is closer to FP32 than HF FP16's on five of the six prompts.

HF FP16 is closer on the other 738 tokens, including the single worst token of row 2802. So the
engine is not better on every measurement.

What remains is the worst case. Some tokens are still far from FP32 in both implementations. The
largest final-state error across the six prompts is 26.6% for the engine and 29.9% for HF FP16.
On row 2802, our engine's worst token was about twice as far off as HF FP16's.

## Next steps

I don't consider this issue fully closed. A worst-case token sitting 26% away from FP32 still bothers
me, and six prompts chosen for their outliers are not a guarantee for every input. But on these
prompts, the engine is closer to FP32 than HF's own FP16 implementation for the vast majority of
tokens, and usually in the tail as well. Row 2802's worst token is the exception. That is good
enough for me to use it as the inference engine for harvesting activations. I'll keep
this as an open issue and come back to the tail when it starts to matter for the analysis.
