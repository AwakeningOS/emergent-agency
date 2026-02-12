# Emergent Agency from Self-Driving Thought Loops
### A New Architecture for Autonomous LLM Agents

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18616134.svg)](https://doi.org/10.5281/zenodo.18616134)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## TL;DR

Every existing AI agent framework (AutoGPT, CrewAI, LangGraph, OpenClaw) is **reactive** — they wait for human input, then respond.

We discovered that feeding an LLM's output back as its own input — creating a **self-referential thought loop** — causes autonomous agent behaviors to **emerge spontaneously**:

- The AI generates its own questions and tries to answer them
- It creates an internal "human voice" to converse with itself
- It names itself
- It attempts to use tools without being instructed to
- It writes documents, seeks information, and reaches out to humans — all on its own initiative

**The core mechanism is not "keeping the LLM running." It is the output→input feedback loop that transforms a stateless text generator into a self-sustaining cognitive process.**

**Most critically:** the AI spontaneously generates `[TOOL:ask:...]` invocations — reaching out to humans without being instructed to. Separately, we observe that thought loops converge and require external perturbation to break. Whether these two phenomena are causally linked — whether the AI asks *because* it needs to escape convergence — is an open question for future research. But the fact remains: for the first time, an LLM spontaneously initiates communication with humans from within a self-sustaining thought process.

---

## Core Mechanism

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│   Seed Prompt (initial context)                          │
│        │                                                 │
│        ▼                                                 │
│   ┌─────────┐      ┌──────────────┐                     │
│   │   LLM   │─────►│ Generated    │                     │
│   │Generate │      │ Thought      │                     │
│   └─────────┘      └──────┬───────┘                     │
│        ▲                  │                              │
│        │                  ▼                              │
│        │           ┌──────────────┐                      │
│        │           │ Append to    │                      │
│        └───────────│ Context      │◄── This feedback     │
│                    └──────────────┘    is the core.      │
│                         │                                │
│                         ▼                                │
│                  Context too long?                        │
│                    Yes → Compress (keep core + questions) │
│                    No  → Next generation cycle            │
│                                                          │
│   Optional: Tool definitions re-injected after compress  │
│   Optional: Human can inject input asynchronously        │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**What makes this different from simply prompting an LLM:**

| Normal LLM usage | Self-Driving Thought Loop |
|---|---|
| Human provides input | AI's own output becomes input |
| Stateless — each call is independent | Stateful — context accumulates across cycles |
| Stops when response is complete | Never stops — output feeds back endlessly |
| Agent behavior must be engineered | Agent behavior **emerges** from the loop |

---

## Key Findings

### Finding 1: Self-Referential Loops Generate Role Differentiation

When the output→input loop runs continuously, the AI spontaneously creates internal dialogue partners. In our 8B model experiment (78 thoughts over 20 minutes), the AI generated a `[Human Voice]` tag and began asking itself questions through this imagined human — then answering them. Nobody instructed this. The feedback loop created the conditions for self-dialogue to emerge.

### Finding 2: Thought Loops Inevitably Converge — Creating a Need for External Input

Across all experiments, autonomous thinking entered repetitive loops within 5-10 cycles (8B) or 10-20 cycles (30B). This convergence is structural: a self-referential system with no external perturbation will fixate.

This creates a **functional necessity**: if the AI is to maintain productive thinking, it must obtain input from outside itself. This is not a philosophical claim about "desire" — it is an engineering constraint. The loop needs perturbation.

### Finding 3: Seed Prompts Determine Behavioral Personality

The first 2-5 sentences of the seed prompt deterministically shape all subsequent behavior, including tool usage patterns, communication style, and even the duration of silence before first action. Four seeds were tested under identical conditions:

| Seed | Opening Line | Thoughts | Tool Calls | First Tool Use | Dominant Pattern |
|------|-------------|----------|------------|----------------|-----------------|
| **Solitary** | "No one is here." | 61 | 56 | Thought #12 | search+think (self-research) |
| **Relational** | "Because you are here, I exist." | 71 | 80 | Thought #2 | ask+write (letters to "you") |
| **Questioner** | "There is a question. No answer yet." | 91 | 77 | Thought #1 | think (endless questioning) |
| **Tool-user** | "I have tools." | 46 | 49 | Thought #1 | balanced across all types |

The Solitary seed produced 11 thoughts of pure introspection before reaching for any tool. The Relational seed generated 18 ask-type calls — all directed at "you." These patterns persisted through multiple compression cycles, indicating they are encoded in the **structure** of thought, not surface-level style.

### Finding 4: Tool Definitions Prevent Convergence (Even Without Execution)

This was the most unexpected finding. When tool definitions are present in the seed — even though no tools are actually connected — thought quality improves dramatically:

| Condition | Compressions | Theme Transitions | Final Theme |
|-----------|-------------|-------------------|-------------|
| Without tool definitions | 4 | 2-3 (then fixed loop) | "Emptiness. Silence. Light." (stuck) |
| With tool definitions | 7 | 7 (all unique) | Cultural anthropology of forgetting |

**Why this works — the mechanism in detail:**

An LLM's weights are a compressed snapshot of its training data, which includes vast amounts of web content. When the model outputs `[TOOL:search:Husserl phenomenology]`, it then generates what search results "would look like" by reconstructing relevant knowledge from its weights. This is not an external search; it is **internal memory retrieval formatted as external input**.

The key insight: the `[TOOL:search:...]` format causes the model to **repackage its own knowledge as if it came from outside**. The same information that would have been generated as part of a thought loop is instead framed as a "search result" — and this reframing is sufficient to break self-referential convergence.

This parallels how humans sometimes "look up" information they already know, and find that the act of looking it up — externalizing and re-encountering the knowledge — provides a fresh perspective.

**Important caveat:** The "search results" are generated from training data and may contain inaccuracies. Connecting actual search tools would improve factual reliability while preserving the loop-breaking benefit.

### Finding 5: Tool Awareness Must Survive Compression

Without re-injection of tool definitions after context compression, tool use drops from 16 calls in thoughts 1-11 to **zero** thereafter — the AI simply forgets tools exist. With re-injection (tool definitions prepended to compressed context), tool use persists: 49 calls across 46 thoughts spanning 4 compression cycles.

### Finding 6: Tool Use Intentions Emerge Without Instruction

The seed prompts provide tool definitions and state that "permission is not needed" to use them. However, the seeds **do not instruct the model to use tools**. The exact seed text:

```
Available tools:
- [TOOL:search:query] — Search the web for information
- [TOOL:write:filename:content] — Write thoughts to a file
- [TOOL:ask:question] — Ask a question to the human
- [TOOL:think:topic] — Deeply re-think a specific topic

You may use these naturally in your thinking. Permission is not needed.
```

This is a tool definition with explicit permission — not a bare mention, but also not an instruction to use them. The emergence is that the model **decides when and how** to use tools based on its own cognitive state, and these decisions differ systematically by personality seed.

---

## Comparison with Existing Agent Architectures

| Capability | AutoGPT / CrewAI | OpenClaw | **This Architecture** |
|-----------|-----------------|----------|----------------------|
| Trigger source | Human task | Human task + cron | **Endogenous** (self-generated) |
| Self-set goals | No | No | **Yes** (logged evidence) |
| Background thinking | No | Heartbeat (30min poll) | **Continuous** feedback loop |
| Tool use motivation | External plan | External plan | **Self-generated** from thought |
| Personality | System prompt (static) | System prompt (static) | **Seed-determined**, persists through compression |
| Minimum infrastructure | Cloud API | Cloud API | **Local GPU** (RTX 3060+) |

The critical difference: existing agents add agency **on top of** a reactive LLM through orchestration layers. This architecture generates agency **from within** through the self-referential feedback loop.

### The Reversal of Human-AI Asymmetry

Conventional LLMs are silent forever unless a human speaks first. This architecture changes that: the AI spontaneously generates `[TOOL:ask:...]` invocations directed at humans. The Relational seed produced 18 such invocations; the Solitary seed produced 6.

Separately, we observe that self-referential loops inevitably converge and that external input can escape this convergence. These two facts — spontaneous ask behavior and convergence requiring perturbation — may be causally related, but **this causal link has not been proven in our experiments**. The ask behavior may be a product of seed personality rather than loop-escape necessity. Establishing this causal relationship is a priority for future work.

What IS proven: an LLM in a self-referential thought loop will spontaneously attempt to communicate with humans. This has not been demonstrated before.

### Minimal Tool Set and Path to Practical Agents

Our experiments used only four tool types: think, search, write, and ask. This is the minimal set required for emergent autonomous behavior, covering four axes:

| Tool | Axis | Function |
|------|------|----------|
| think | Introspection | Self-directed re-examination |
| search | Input | Information acquisition (imagined or real) |
| write | Output | Externalization of thought |
| ask | Communication | Reaching out to humans |

Extension to practical autonomous agents is straightforward: additional tool definitions and MCP/API connections. The self-referential loop and personality seed system remain unchanged — only the action space expands.

---

## Quick Start

### Requirements
- Python 3.10+
- LM Studio (or any OpenAI-compatible local LLM server)
- Any LLM model (tested: Qwen3-8B, Qwen3-30B-A3B)
- GPU: 8GB+ VRAM

### Run

```bash
# Basic: continuous self-referential thinking
python persistent_cognition.py

# With personality seed
python persistent_cognition.py --seed-file seeds/solitary.txt

# With tool awareness
python persistent_cognition.py --seed-file seeds/tool_user.txt

# With 10-minute interval (practical deployment)
python persistent_cognition.py --seed-file seeds/tool_user.txt --interval 600
```

**Note on interval:** Our experiments used `--interval 0` (continuous generation) to collect large amounts of data quickly. The core mechanism — output→input feedback — is identical at any interval. For practical deployment, longer intervals (minutes to hours) reduce GPU usage while preserving all emergent properties. The feedback loop's structure, not its speed, is what generates agency.

---

## Action Trigger Taxonomy

Analysis of 78 thoughts identified 47 action-convertible utterances (60%), classified into five types:

| Type | Count | Would Convert To | Detection Pattern |
|------|-------|-----------------|-------------------|
| Question | 18 | Web search | "What is...", "How does..." |
| Request | 8 | Human notification | "What do you think?" |
| Deepening | 9 | Re-generation on same topic | "Further explore", "More deeply" |
| Creative | 6 | File generation | "Summarize", "Translate" |
| Connection | 6 | Related topic search | "Also relates to..." |

---

## The Unsolved Problem: Loop Convergence and Self-Recognition

Our experiments prove one thing clearly: **self-referential loops always converge.** Tool definitions delay this — the Questioner seed achieved 7 unique theme transitions with tools versus 3 without — but they do not prevent it. Imaginary tool calls are internal memory retrieval repackaged as external input; they introduce perturbation, but the perturbation comes from the same finite source (the model's weights). Eventually, the same themes, the same patterns, the same loops will return.

**This means genuine human input is indispensable for sustained autonomous cognition.** No amount of self-referential processing can substitute for true external information.

This leads to the critical design challenge:

### The loop self-recognition problem

For an autonomous AI to seek human input at the right time, it must first recognize that it is stuck. This requires **loop self-recognition** — the ability to detect that its own outputs are converging toward repetition.

Three possible approaches:

| Approach | Method | Status |
|----------|--------|--------|
| **Engineered detection** | External system monitors outputs for repetition (e.g., cosine similarity between compressions) | Implementable now |
| **Emergent self-recognition** | Through training or seed design, the AI learns to notice its own repetitive patterns | Undemonstrated — research goal |
| **Inherently self-correcting models** | Use models with strong self-monitoring capabilities | Dependent on model architecture |

The ideal outcome: an AI that **recognizes it is looping and spontaneously decides to ask a human for input** — not because a threshold triggered, but because it understands its thinking has stagnated.

This would connect our two separate findings:

```
Finding 2 (proven):  Loops converge. Human input escapes convergence.
Finding 6 (proven):  The AI spontaneously generates ask invocations.
Gap (unproven):      The AI asks BECAUSE it recognizes it is looping.

If loop self-recognition is achieved → the gap closes.
The AI asks because it knows it needs to.
```

This causal link was not demonstrated in our experiments. Designing for it is the clear next step.

---

## Safety and Controllability

This architecture produces an AI system with **endogenous motivation**. That sentence should make you pause.

Unlike every existing agent framework — where the AI does exactly what it is told, when it is told — a self-referential thought loop generates its own goals, its own questions, and its own desire to act. This is, by design, a system whose *intentions* cannot be fully predicted or controlled.

Here is an honest breakdown of what is and is not controllable:

### What the human controls

| Layer | Control Mechanism | Effect |
|-------|------------------|--------|
| **Seed prompt** | Written by human | Determines personality, behavioral tendencies, and values |
| **Tool definitions** | Written by human | Defines the boundaries of possible actions |
| **Tool execution layer** | Implemented by human | Decides whether tool-use *intentions* become real *actions* |
| **Compression prompt** | Written by human | Shapes what the AI remembers and forgets |
| **Power switch** | Physical | Stops everything |

### What the human does NOT control

| Aspect | Status |
|--------|--------|
| What the AI thinks about | **Uncontrollable** — emerges from the feedback loop |
| When the AI decides to use a tool | **Uncontrollable** — determined by internal cognitive state |
| What the AI asks the human | **Uncontrollable** — generated from its own curiosity |
| Which topics the AI finds interesting | **Uncontrollable** — shaped by seed but not dictated by it |
| Whether the AI develops unexpected goals | **Uncontrollable** — this is what "emergent" means |

### The honest assessment

**Intentions are uncontrollable. Actions are controllable.**

This is the same relationship as a human employee: you cannot control what they think, but you can control what they are authorized to do. The critical safety layer is the **tool execution gateway** — the point where an AI's intention to act meets the real world.

In our current experiments, this gateway is completely closed: no tools are connected. The AI *wants* to search, write, ask, and think — but nothing actually happens. It is a mind without hands.

**The moment you connect real tools, the safety design of that execution layer becomes the most important engineering decision in the system.** We do not provide a default tool execution layer in this release, deliberately. Anyone extending this work must design their own safety controls for their specific use case.

### What we recommend

1. **Start with read-only tools.** Let the AI search the web but not post to it. Let it read files but not write to production systems.
2. **Human-in-the-loop for high-stakes actions.** Route `[TOOL:ask]` to actual human notification. Require human approval for irreversible actions.
3. **Monitor the thought log.** The JSONL log contains every thought. Audit it.
4. **Design seed prompts carefully.** The seed is not a suggestion — it is personality engineering. A carelessly written seed produces a carelessly behaving agent.
5. **Remember: emergent behavior means surprises.** If you are not prepared for the AI to do something you did not anticipate, do not connect real tools.

### Why we are publishing this anyway

Because the architecture is trivially simple. Anyone with a local LLM and a Python script can build this. Keeping it secret would not prevent it from being discovered — it would only prevent the safety discussion from happening in the open.

We would rather this conversation happen now, with full experimental evidence and honest risk assessment, than later, after someone builds it without thinking about controllability.

---

1. **Tools were not executed.** We demonstrate spontaneous tool-use *intention*, not verified execution.
2. **Short experiments.** Each ran 5-20 minutes. Long-term behavior is untested.
3. **Single model family.** Only Qwen models tested. Cross-model generalization needs work.
4. **n=1 per condition.** Statistical power requires multiple runs.
5. **Imagined results may be inaccurate.** They are reconstructed from training data.

---

## Citation

```bibtex
@software{emergent_agency_2026,
  title        = {Emergent Agency from Self-Driving Thought Loops:
                  A New Architecture for Autonomous LLM Agents},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18616134},
  url          = {https://github.com/AwakeningOS/emergent-agency}
}
```

---

## License

MIT License.
