#!/usr/bin/env python3
"""Generate the fixed 96 x ~200-word prompt set for the 1.2 measurement.

Deterministic (fixed seed); the generated JSONL is committed and its sha256 is
pinned in every run manifest. Each prompt carries a unique prefix so cold-start
levels satisfy the contract's unique-prefix rule.
"""

import hashlib
import json
import random
from pathlib import Path

N_PROMPTS = 96
TARGET_WORDS = 200
SEED = 20260727

TOPICS = [
    "the history of container shipping", "how photosynthesis works",
    "the design of suspension bridges", "the economics of public transit",
    "how vaccines train the immune system", "the physics of sailing upwind",
    "the evolution of writing systems", "how weather forecasting models work",
    "the chemistry of baking bread", "the architecture of medieval castles",
    "how GPS satellites determine position", "the ecology of coral reefs",
    "the invention of the printing press", "how batteries store energy",
    "the mathematics of compound interest", "the formation of river deltas",
]

FILLER_SENTENCES = [
    "Explain the key mechanisms involved and why they matter in practice.",
    "Describe the historical context that shaped its early development.",
    "Compare the main competing approaches and their trade-offs.",
    "Discuss the physical principles that constrain what is possible.",
    "Outline the sequence of steps from raw inputs to finished outcome.",
    "Identify the common misconceptions people hold about this subject.",
    "Summarize the measurable quantities an engineer would track here.",
    "Consider how scale changes the behavior of the overall system.",
    "Note the failure modes that occur when conditions are pushed to extremes.",
    "Mention the tools and instruments used to study this area today.",
    "Address how costs and benefits are typically weighed by practitioners.",
    "Relate the topic to an everyday experience a reader would recognize.",
    "State which open questions remain unresolved among specialists.",
    "Trace how improvements in materials changed what could be built.",
    "Evaluate the environmental considerations that influence design choices.",
    "Reflect on how the field is likely to change over the next decade.",
]


def build_prompt(rng: random.Random, idx: int) -> str:
    topic = TOPICS[idx % len(TOPICS)]
    head = (
        f"Request {idx:03d} of the fixed benchmark prompt set. "
        f"Write a clear, well-organized explanation of {topic}."
    )
    words = head.split()
    while len(words) < TARGET_WORDS:
        s = rng.choice(FILLER_SENTENCES)
        words.extend(s.split())
    return " ".join(words[:TARGET_WORDS])


def main() -> None:
    rng = random.Random(SEED)
    out = Path(__file__).parent / "prompts_200w_96.jsonl"
    with out.open("w") as f:
        for i in range(N_PROMPTS):
            prompt = build_prompt(rng, i)
            f.write(json.dumps({"prompt_id": i, "prompt": prompt,
                                "n_words": len(prompt.split())}) + "\n")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"wrote {out} ({N_PROMPTS} prompts, {TARGET_WORDS} words each)")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
