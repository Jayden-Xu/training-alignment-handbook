"""A fixed prompt set: 64 prompts, 4 to 700 tokens, built from a seeded word list (no download).

Prompts are token-id lists for the Qwen2.5 tokenizer. The engine tokenizes the text once and
records the ids, so the trainer side never needs the tokenizer.
"""
from __future__ import annotations

import random

LENGTHS = [4, 9, 17, 31, 48, 75, 120, 190, 260, 400, 560, 700]

_STARTS = [
    "Explain why the sky is blue.", "Write a short poem about rivers.", "List three uses of copper.",
    "What is the capital of Australia?", "Translate to French: the cat sleeps on the warm chair.",
    "Summarize the following notes.", "Continue the story.", "Answer the question about the table below.",
    "def fibonacci(n):", "The following is a log of a long meeting.", "1, 1, 2, 3, 5, 8,",
    "Dear team, here is the weekly update.",
]
_WORDS = ("time year people way day man thing woman life child world school state family student group "
          "country problem hand part place case week company system program question work government number "
          "night point home water room mother area money story fact month lot right study book eye job word "
          "business issue side kind head house service friend father power hour game line end member law car "
          "city community name president team minute idea kid body information back parent face others level "
          "office door health person art war history party result change morning reason research girl guy "
          "moment air teacher force education river copper market engine rain bridge garden signal").split()


def prompt_texts(n: int = 64, seed: int = 0):
    """n (text, target_tokens) pairs. Text is long enough; the engine cuts ids to target_tokens."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        target = LENGTHS[i % len(LENGTHS)] + rng.randrange(0, 4)
        words = [_STARTS[i % len(_STARTS)]]
        while len(words) < 2 * target + 8:
            sent = " ".join(rng.choice(_WORDS) for _ in range(rng.randrange(5, 14)))
            words.append(sent.capitalize() + rng.choice([".", ",", ";", "?", "."]))
        out.append((" ".join(words), target))
    return out


def prompt_ids(tokenizer, n: int = 64, seed: int = 0):
    ids = []
    for text, target in prompt_texts(n, seed):
        t = tokenizer.encode(text, add_special_tokens=False)
        ids.append([int(x) for x in t[:target]])
    return ids
