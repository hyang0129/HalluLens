# First-token token-wise contrastive learning

Status: issue #151 implementation draft. The 50-cell queue builder is included;
workers are intentionally not started by this change.

## Motivation

The working hypothesis is that hallucination risk is substantially determined
before the response unfolds. An autoregressive language model with fixed
parameters defines a conditional distribution over complete continuations as
soon as it receives the prompt:

\[
p_\theta(Y_{0:T-1}\mid X)
=
\prod_{t=0}^{T-1}
p_\theta(Y_t\mid X,Y_{<t}).
\]

This means the distribution over possible responses exists at decoding step
zero, before a response token is sampled. It does **not** imply that the first
token, its logits, or one hidden vector explicitly contains every future
conditional distribution. Those conditionals still branch on possible token
histories, and the complete model state includes the parameters and KV cache.
The defensible claim is therefore:

> The first decoding state may be an approximately sufficient statistic for
> predicting whether the eventual response will hallucinate.

This is an empirical sufficiency hypothesis, not a consequence guaranteed by
the chain rule. A narrower theorem does hold for the benchmark's substring
label at the **distribution** level; the unproven part is whether the captured
activation stack is sufficient to recover that distributional functional.

### Capture semantics

For the current `icr_capture` format, response activation position `q=0` is
already the correct pre-response surface. Hugging Face generation returns the
prefill hidden states at step zero; HalluLens stores the final prompt position
at every layer as `response_activations[:, 0, :]`. Those are the hidden states
whose logits produced response token zero. They do not condition on the sampled
first response token. At `q>=1`, the stored state is the decode pass that
produced token `q` and therefore conditions on response tokens `<q`. See
[`activation_logging/generate_capture.py`](../../activation_logging/generate_capture.py).

Accordingly, “first token” below means **the decoding state that predicts the
first generated token**, not a representation computed after reading that
token.

## Information-theoretic statement

Let:

- `X` be the prompt observed by the language model;
- `A` be the external answer/reference used to evaluate the response;
- `Y` be the generated response;
- `C = c(Y,A) in {0,1}` be the hallucination label of `Y`;
- `S_t = (h_t^1, ..., h_t^L)` be the stack of hidden states across model
  layers at decoding step `t`;
- `Z_t = f_phi(S_t)` be the learned 512-dimensional detector representation.

The strong first-state sufficiency condition would be

\[
C \perp S_{1:T-1}\mid S_0,
\]

or equivalently

\[
I(C;S_{0:T-1}) - I(C;S_0) = 0.
\]

We do not assume exact equality. The experimental hypothesis is that the
incremental information is small enough for detection:

\[
I(C;S_{0:T-1}) - I(C;S_0) \approx 0.
\]

Since `Z_0` is a compressed function of `S_0`, the data-processing inequality
gives

\[
I(C;Z_0) \le I(C;S_0).
\]

The encoder cannot create information about hallucination that is absent from
the first decoding state. Its role is to discard nuisance variation while
retaining the portion predictive of `C`, which is the information-bottleneck
interpretation of the proposed contrastive objective.

### Substring-match theorem for the 64-token benchmark

The default benchmark generation is explicitly capped at 64 new tokens and
the default label is a case-insensitive substring match against the gold answer
or any accepted alias. The task prompts also ask the model to answer concisely.
This permits an exact benchmark-specific result.

Let `A={a_1,...,a_m}` be the normalized accepted-answer strings. For a decoded
token sequence `Y`, define

\[
M_r(Y,A)
=
\mathbf{1}\{\text{some }a_j\in A
\text{ is a substring of decode}(Y_{0:r-1})\},
\]

and define the benchmark hallucination label

\[
C_{64}=1-M_{64}(Y,A).
\]

**Theorem 1 — decode-start determinacy.** For fixed model parameters, prompt,
answer set, tokenizer, and decoding rule, the first-step joint continuation
distribution uniquely determines the exact probability of the benchmark label:

\[
\rho_{64}(X,A)
=
P_\theta(C_{64}=1\mid X,A)
=
1-
\sum_{y_{0:63}}
M_{64}(y,A)
P_\theta(y_{0:63}\mid X).
\]

**Proof.** `C_64` is a deterministic measurable function of a generated
sequence of at most 64 tokens and the fixed answer set. The autoregressive
factorization at decoding step zero assigns a probability to every such
sequence, including sequences terminated early by EOS. Summing the probability
mass of sequences for which the substring predicate is one gives the match
probability; its complement is the hallucination probability. No generated
token needs to be observed to define this risk. `□`

Under deterministic greedy decoding, the distribution is a point mass, so
`C_64` itself—not merely its probability—is fixed at decoding step zero.

This theorem is exact for the current 64-token output contract. If the intended
claim concerns an uncapped response, an explicit early-answer assumption gives
the required approximation bound. Let `E_r={M_r=1}` and let `E_infinity` be the
event that an accepted answer appears anywhere in the eventual response.
Assume the direct-answer prompt induces

\[
P(E_{64}\mid E_{\infty},X,A)\ge 1-\epsilon.
\]

**Theorem 2 — early-answer truncation bound.** Under this assumption,

\[
0
\le
P(E_{\infty}\mid X,A)-P(E_{64}\mid X,A)
\le
\epsilon,
\]

and therefore the 64-token and eventual substring-hallucination risks differ
by at most `epsilon`.

**Proof.** Since `E_64` is a subset of `E_infinity`,

\[
P(E_{\infty})-P(E_{64})
=P(E_{\infty}\cap E_{64}^{c})
=P(E_{\infty})P(E_{64}^{c}\mid E_{\infty})
\le \epsilon P(E_{\infty})
\le \epsilon.
\]

Taking complements gives the same absolute bound for hallucination risk. `□`

Mere support is insufficient: `P(E_infinity)>0` can be arbitrarily small and
does not imply that a match is likely. The theorem requires the conditional
early-answer mass assumption above. The concise-answer prompt makes a small
`epsilon` plausible, but it must be measured by generating beyond 64 tokens and
recording the first accepted-substring position. Existing 64-token captures
cannot estimate this tail probability because they terminate at the boundary.

### From the theorem to the recorded first state

The theorems concern the full continuation distribution at decoding start. To
make the first-state detector a corollary, one additionally needs

\[
P_\theta(Y_{0:63}\mid X)
=
P_\theta(Y_{0:63}\mid S_0).
\]

**Corollary — first-state risk sufficiency.** If the equality above holds, then
there exists a deterministic function `g` such that

\[
\rho_{64}(X,A)=g(S_0,A).
\]

Consequently, later generated tokens and later activation states are not
needed for the Bayes-optimal prediction of the 64-token substring-hallucination
risk. Under deterministic decoding, `g(S_0,A)` reduces to the binary benchmark
label. This is the formal version of the “only the first state is necessary”
claim, conditional on `S_0` being continuation-sufficient.

The complete transformer decoding state—model parameters, decoding rule, and
prefill KV cache—is sufficient by construction. HalluLens stores only the
final-position hidden state across layers, not the full KV cache, so sufficiency
of the recorded `S_0` remains the substantive empirical hypothesis. Token-wise
contrastive learning is an estimator for the risk functional established by
Theorem 1; it is not the proof that this compressed observation retains it.

There is also an important stochastic-generation qualification. Before
sampling, the appropriate target is hallucination **risk**,
`p(C=1 | S_0)`. If the same prompt can yield truthful and hallucinated samples,
no deterministic first-state detector can know which sampled outcome will
occur. Greedy decoding removes that sampling uncertainty, but it still does not
prove that the recorded cross-layer stack is a sufficient statistic for the
whole model state.

## Geometry: transpose layers and tokens

The existing contrastive dataset produces two layer views. Each view fixes one
model layer and contains the response-token sequence:

\[
V^{\text{layer}}_{i,a}
=
[h^{\ell_a}_{i,0}, ..., h^{\ell_a}_{i,R_i-1}]
\in \mathbb{R}^{R_i\times d}.
\]

The proposed token-wise dataset transposes those axes. Each view fixes one
decoding step and contains the full model-depth sequence:

\[
V^{\text{token}}_{i,a}
=
[h^1_{i,t_a}, ..., h^L_{i,t_a}]
\in \mathbb{R}^{L\times d}.
\]

Two views of the same response use distinct decoding steps `k != n`. The same
encoder processes both layer stacks. This teaches the representation to retain
response-level hallucination information that persists across generation time,
while discarding token-specific surface variation.

For Llama 3.1 8B, the recommended input uses all 32 post-block residual states
and excludes the embedding row in the primary configuration. Including the
embedding row is an ablation.

## Encoder

The sequence axis becomes model depth rather than response time, but the
existing progressive compressor can otherwise be reused:

```text
one decoding step across all layers: (batch, 32 layers, 4096)
    -> fixed layer-depth positional encoding
    -> 4096 -> 2048 + Transformer block
    -> 2048 -> 1024 + Transformer block
    -> 1024 ->  512 + Transformer block
    -> mean pool across layers
    -> 512-dimensional normalized representation
```

Reusing `LogprobReconProgressiveCompressor` unchanged makes this an exact
parameter match to the current single-encoder contrastive model because its
parameters do not depend on sequence length or on what the sequence axis
represents:

| Component | Parameters |
|---|---:|
| Progressive encoder | 77,390,336 |
| Logprob reconstruction decoder | 147,776 |
| Total | 77,538,112 |

The first, second, and third progressive blocks contain 58,753,024,
14,696,448, and 3,678,208 parameters respectively; the final projection
contains 262,656. A new larger architecture is therefore unnecessary for the
first experiment. The parameter-controlled test should change the data
geometry, not capacity.

The positional encoding now denotes layer depth. Fixed sinusoidal encodings
preserve the exact parameter count. Learned layer embeddings can be evaluated
later as a small ablation.

## Token-pair sampling

For response `i` with captured length `R_i >= 2`, sample two distinct decoding
steps and construct

\[
(V_{i,1}, V_{i,2}) = (S_{i,k}, S_{i,n}),\qquad k\ne n.
\]

The recommended primary schedule is **first-anchored**:

```text
k = 0
n ~ Uniform({1, ..., min(R_i, R_max) - 1})
```

This exposes the first decoding state on every update and explicitly transfers
response-level signal from later states into the deployment representation.
The pure token-invariance variant samples both `k` and `n` uniformly without
replacement. A mixed schedule can use first-anchored pairs half the time and
fully random distinct pairs half the time.

Responses with fewer than two captured steps cannot satisfy `k != n`. Keep
them in the token-zero evaluation/reference surfaces, but exclude them from
pair-training and pair-loss validation datasets. A same-step pair created only
through dropout is a separate ablation and should not be silently mixed into
the primary condition.

## Training objective

For each example, encode both layer-stack views with shared weights:

\[
z_{i,k}=\operatorname{norm}(f_\phi(S_{i,k})),\qquad
z_{i,n}=\operatorname{norm}(f_\phi(S_{i,n})).
\]

Use the same supervised contrastive semantics as the current standard
HalluLens encoder:

- the two token positions from the same response are always a positive pair;
- with `ignore_label=1`, truthful responses also form cross-example class
  positives;
- hallucinated responses retain their within-response token positive but repel
  other examples, including other hallucinated examples;
- other batch examples provide negatives.

The primary loss is

\[
\mathcal{L}
=
\mathcal{L}_{\mathrm{SupCon}}
\left(\{z_{i,k},z_{i,n}\}, C; \texttt{ignore\_label}=1\right)
+
\lambda_{\mathrm{recon}}\mathcal{L}_{\mathrm{recon}}.
\]

The existing auxiliary decoder can ask each single-step representation to
reconstruct the full generated-token logprob trajectory. For a pair `(k,n)`,

\[
\mathcal{L}_{\mathrm{recon}}
=
\tfrac{1}{2}
\left[
\operatorname{MSE}(g(z_{i,k}),\ell_i)
+
\operatorname{MSE}(g(z_{i,n}),\ell_i)
\right],
\]

with the existing NaN mask, variance suppression, and fixed-length resampling.
This auxiliary is conceptually aligned with the hypothesis: a representation
from one decoding step is trained to retain sequence-level uncertainty. It is
not evidence that the first state is sufficient, so `lambda_recon=0` must be a
reported ablation.

No binary classifier is required in the primary experiment. KNN on the learned
representation remains the headline score, preserving comparability with the
existing contrastive model. A frozen linear probe can be reported as a
secondary scoring ablation.

## Dataset and implementation contract

The existing memmap layout already contains the necessary tensor:

```text
response_activations: (sample, layer + embedding, decoding_step, hidden_dim)
```

A new token-wise dataset view should emit:

```text
views_activations:  (2, selected_layers, hidden_dim)
view_token_indices: (2,)  # [k, n]
halu:               scalar
response_len:       scalar
logprob:            existing full response target
```

Important invariants:

1. Select token positions only from real decoding steps `< response_len`.
2. Use the same `(k,n)` across every selected layer for one view pair.
3. Use all post-block layers in fixed depth order; never randomly permute them.
4. Do not feed token IDs, the sampled first token, response length, or absolute
   token position into the primary encoder.
5. Training and evaluation splits, labels, seeds, temperature, batch size,
   optimizer, reconstruction weight, and total update count must match the
   current contrastive baseline.
6. At evaluation, build both the train reference bank and test embeddings from
   `t=0` only. Do not compare a random-token training bank to first-token test
   embeddings.
7. MMLU remains outside the benchmark suite.

The trainer can largely remain unchanged because it already expects
`(batch, num_views, sequence, hidden_dim)`. Only the meaning of `sequence`
changes from response tokens to model layers, and `view_indices` changes from
layer IDs to decoding-step IDs.

## Evaluation plan

### Primary comparison

Compare, with identical datasets and five seeds:

1. current layer-wise encoder evaluated at response prefix `k=1`;
2. token-wise encoder trained with first-anchored `(0,n)` pairs and evaluated
   at decoding step zero;
3. token-wise encoder trained with fully random distinct `(k,n)` pairs and
   evaluated at decoding step zero;
4. full-response current contrastive encoder as the information-rich reference.

Report AUROC and AUPRC on HotpotQA, NQ, PopQA, SciQ, and SearchQA. For the
token-wise methods, report KNN as primary and a frozen linear probe only as a
diagnostic.

### Necessary ablations

- first-anchored versus random-distinct token pairs;
- full logprob reconstruction versus `lambda_recon=0`;
- all 32 post-block layers versus the current selected-layer subset;
- standard supervised contrastive labels versus label-free SimCLR;
- fixed layer-depth encoding versus no depth encoding;
- evaluation at `t=0,1,3,7,15,31,63` where available, as a diagnostic curve.

The later-token curve is not the headline. It measures how much new
hallucination information appears after generation begins:

\[
\Delta_t = \operatorname{AUROC}(Z_t)-\operatorname{AUROC}(Z_0).
\]

Large positive `Delta_t` falsifies the strong first-state sufficiency story.

## Success and falsification criteria

The approach is supported if the parameter-matched token-wise encoder improves
first-state AUROC consistently over the layer-wise `k=1` encoder and closes a
meaningful fraction of the gap to full-response detection. Similar performance
at later token positions would support token invariance.

The approach is weakened or falsified if:

- later states substantially outperform `t=0`;
- first-anchored pairing does not improve `t=0` over random pairing;
- token-wise alignment causes representation collapse or erases useful
  time-specific information;
- gains disappear without full-response logprob reconstruction, indicating
  that the auxiliary target rather than token-wise contrast is doing the work;
- performance depends on later-token reference embeddings at KNN evaluation.

The strongest defensible paper claim, if successful, is:

> For a fixed 64-token substring-match benchmark, hallucination risk is an exact
> functional of the continuation distribution defined at decoding step zero.
> Under an explicit early-answer assumption, the same risk approximates
> eventual substring correctness within `epsilon`. We test whether a compressed
> cross-layer observation of the first decoding state retains enough information
> to estimate that functional. Token-wise contrastive training makes this
> representation hypothesis operational by aligning the first state with later
> states from the same response.

## Conceptual references

- Claude Shannon, [A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf), 1948 — entropy, mutual information, and the data-processing principle.
- Naftali Tishby, Fernando Pereira, and William Bialek, [The Information Bottleneck Method](https://arxiv.org/abs/physics/0004057), 1999/2000 — compressed representations that preserve target-relevant information.
- Yoshua Bengio et al., [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/v3/bengio03a.html), 2003 — sequence probability expressed as a product of next-token conditional probabilities.
