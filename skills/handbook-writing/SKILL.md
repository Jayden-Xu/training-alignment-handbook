---
name: handbook-writing
description: Write or rewrite a page of the Training Alignment Handbook as a teaching page about a workflow, so an engineer understands why each step exists and how to do it, without reading experiment logs. Use for any new page, any rewrite, and before adding a measured result.
---

# Writing a handbook page

The handbook teaches three workflows: finding where two pre-training runs part
(part 1), making a rollout engine and a trainer agree (part 2), and making one
operator invariant (part 3). Part 0 teaches the ideas all three rest on. Each
page is one step or one idea of a workflow. Experiments are evidence and live
in reports, not on the page.

## Who reads it

An ML systems engineer who knows PyTorch, GPUs and RL training and has a
problem now. They read a page to learn what to do next and why.

## How a page is built (the d2l order)

1. **Where we are.** One or two sentences linking to the previous step: what
   the reader has now and what is still wrong. The first page of a part says
   what the whole workflow achieves and lists its steps.
2. **The idea, by intuition first.** Explain the mechanism with a tiny example
   or a picture before any rule: three numbers added in two orders; one row in
   a batch of 1 and of 64; a decode step against a prefill. Then state the
   rule in one sentence and name the general principle it instances.
3. **The step.** What to do, in order. For each action: its goal, how (the
   tool or skill, one line, linked), and what success looks like. Options as
   a small table: option, what it gives, cost class, when to choose it.
4. **What can go wrong.** Pitfalls as `!!! warning` admonitions: one sentence
   of cause, one of consequence, one of what to do.
5. **Summary.** Three to five bullets the reader should remember.
6. **Go deeper.** The report with the measurements, the code, the skill.

## What the page does not contain

- **No specific experiments.** No GPU model, driver, library version, model
  size, batch size sweep, timing, percentage of elements, or count of checks.
  Those are in `examples/<name>/REPORT.md`. The page states the general
  behaviour ("cuBLAS can pick a different kernel when the number of rows
  changes, so the same row can round differently in a different batch") and
  links the report as evidence ("measured: see report").
- No description of how an agent built something, no retries, no file-by-file
  inventory.
- No code identifiers in explanatory sentences. Tool and skill names appear
  only in "The step" and "Go deeper".

Cost is described by class, not by number: "free", "small (a few percent)",
"moderate (under 2x)", "large (2x or more)", and the report has the numbers.
A claim of what holds (bitwise equal, invariant at L2) may be stated, with a
link to its evidence and the conditions it was shown under in plain words
("shown on one GPU type", "shown up to 16 GPUs on two nodes").

## Sentences

- One idea per paragraph; start with it.
- Define each term at first use or link the glossary; use the same word for
  the same thing everywhere.
- Prefer a figure (Mermaid) or a small table to a paragraph of structure.
- No sales words, no "simply", no hedging chains.

## Honesty

- Never weaken a result in words: bitwise means bitwise; a known-answer
  demonstration is called one; what was emulated is called emulated.
- Untested matters stay visible: a short "Not yet shown" line where it changes
  what a reader would do, and the full list in open-items.

## Chinese pages

Write the Chinese page as a page, not a sentence-by-sentence translation: same
sections, links and code; natural Chinese order; the glossary's English terms
kept. Short sentences.

## Checklist

- [ ] The page opens by saying where the reader is in the workflow.
- [ ] The mechanism is explained with a tiny example before any rule.
- [ ] No hardware model, version, size, timing or percentage on the page.
- [ ] Every action names its tool or skill and its success condition.
- [ ] Every claim of what holds links to its evidence.
- [ ] en and zh have the same sections, links and code (`scripts/check_i18n.py`).
- [ ] `mkdocs build --strict` passes.
