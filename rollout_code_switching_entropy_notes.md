# Code-switching and entropy in training rollouts

## Run and comparison

The audit covered 7,168 Nemotron 1.5B mAceReason math training rollouts from steps 1–56, split evenly between English and Bengali prompts. The run used a learning rate of `3e-6` with reference KL disabled. The table compares steps 1–10 with steps 47–56. [Rollout dumps](checkpoints/macereason-math-1500x2-en-bn-nemotron-1.5b-lr3e-6-no-kl-en/rollouts)

| Metric | English prompts | Bengali prompts |
| --- | ---: | ---: |
| Boxed-answer accuracy | 78.4% → 49.8% | 65.0% → 37.7% |
| Same boxed expression appears 3+ times anywhere | 64.4% → 95.9% | 60.5% → 96.4% |
| Same boxed expression appears 3+ times after `</think>` | 11.9% → 32.0% | 14.2% → 47.7% |
| At least 12 Han characters after `</think>` | 2.0% → 3.1% | 7.0% → 17.7% |

Repetition increased in both languages. The increase in Han-script text was larger on Bengali prompts. Fewer than 1% of the late-window rollouts opened with Han characters, so most of the observed switching occurred within responses. Han-character counts are a proxy for Chinese-script text.

Mean actor entropy rose from **1.05 to 1.90** while mean response length fell from **3,257 to 2,121 tokens**. The snapshot had no rollouts flagged as truncated. The early and late windows contained different training problems, so the training accuracy comparison is descriptive. On the fixed validation set, step 50 reached **49.6% mean accuracy across four samples**. [Training log](.slurm/mace500-sdpo-8165053.out)

## Interpretation

Reference selection was a plausible route for SDPO to reinforce repetitive or off-language responses. The reward checked the boxed answer, and the run used prompt language to identify English references. In steps 47–56, **29.1% of all English-prompt rollouts** repeated a boxed expression at least three times somewhere in the response and were still eligible as references. The run stripped thinking before constructing teacher prompts, so the eligibility count cannot establish how much repetitive text reached the teacher. [Reference selection](verl/trainer/ppo/ray_trainer.py)

Actor entropy averages next-token entropy over the sampled rollout prefixes, which changed as responses became shorter and more repetitive. The audit establishes that entropy rose alongside these changes. Whether it rose on fixed prefixes, or contributed causally to code-switching, remains open.

## Bengali continuations with and without an English reference

The generation probes compared the pretrained Nemotron 1.5B base model
(the fixed reference policy used for KL when enabled) against the step-50
SDPO checkpoint from the run above. Each condition generated three fresh
responses for each of 11 Bengali training problems, for 33 responses per
model and condition. The problems came from steps 48–56 and had both a
failed Bengali rollout and an eligible English success. Selection favored
failed rollouts with Han text in the answer, so these counts describe a
targeted probe. [Case selection](../../sdpo-topk-step50/probe.py) · [Generation script](../../sdpo-topk-step50/generate_translation.py)

### Matched English and Bengali worked references

The no-reference condition and both worked-reference conditions used the same
Bengali reminder. The two worked references used equivalent equations and
conclusions, including the final answer. Script and language counts describe
the answer after removing the thinking trace and stripping common LaTeX
math spans and commands. The probes flagged Bengali and Han prose at 12
characters and English prose at five common English function words. These
flags can overlap. “Clean Bengali” means the Bengali flag fired and the
English and Han flags did not. [Matched references](../../sdpo-topk-step50/paired_refs.py) · [Scoring script](../../sdpo-topk-step50/summarize_generation.py)

| Model | Reference context | Correct | English prose | Han text | Clean Bengali | Median tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Base / fixed reference | None | 24/33 | 23/33 | 1/33 | 2/33 | 3,622 |
| Base / fixed reference | English worked solution | 29/33 | 26/33 | 0/33 | 0/33 | 2,800 |
| Base / fixed reference | Bengali worked solution | 26/33 | 23/33 | 1/33 | 1/33 | 3,259 |
| SDPO step 50 | None | 12/33 | 17/33 | 5/33 | 0/33 | 2,091 |
| SDPO step 50 | English worked solution | 33/33 | 25/33 | 0/33 | 1/33 | 1,675 |
| SDPO step 50 | Bengali worked solution | 33/33 | 31/33 | 1/33 | 0/33 | 1,624 |

[Base results](../../sdpo-topk-step50/translation-8180932-summary.txt) · [Step-50 results](../../sdpo-topk-step50/translation-8180856-summary.txt)

Without a reference, the step-50 checkpoint produced more Han-flagged
answers than the base model (5 vs. 1) and fewer correct answers (12 vs. 24).
Adding the English worked reference reduced Han flags to zero for both
models and increased English prose flags, from 23 to 26 for the base and
from 17 to 25 for step 50. The Bengali version still yielded English prose
in 31 of 33 step-50 answers. In this probe, matching the reference language
to the prompt did little to restore Bengali output. The worked references
included the correct answer, so the accuracy gains measure performance with
that information available.

### English reasoning without the final response

A separate probe supplied the English rollout's reasoning before its
closing thinking tag and added a Bengali reminder. The base and step-50
models both scored 33/33. English prose appeared in 9/33 base answers
and 27/33 step-50 answers. Han text appeared in 0/33 and 2/33, respectively,
and neither model produced a clean Bengali answer. The extraction dropped
the final response but retained any answer cues inside the reasoning, so
33/33 does not establish accuracy with the answer withheld. [Reasoning prompt](../../sdpo-topk-step50/generate_reasoning.py) · [Reasoning results](../../sdpo-topk-step50/reasoning-8180634-summary.txt)

An earlier continuation probe used the original Bengali prompt or appended
the actual English rollout as a full example or final-answer-only example.
For the base and step-50 models, respectively, Han flags fell from 3/33
and 4/33 with the original prompt to 0/33 and 0/33 with the full example,
and to 2/33 and 1/33 with the answer-only example. Every English-reference
condition in that probe yielded zero clean Bengali answers. [Earlier continuation results](../../sdpo-topk-step50/generate-8180451-summary.txt)

These continuation probes distinguish Han-script intrusions from English
prose in Bengali answers. English references generally suppressed the
former while English output remained common. The probes sampled new
responses from each prompt, so they leave the role of entropy in the
observed switching unresolved.
