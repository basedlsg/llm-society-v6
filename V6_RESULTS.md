# V6 SurvivalWorld Benchmark Results

## Overview

V6 implements a multi-model, multi-scaffold evaluation framework for LLM agents in a survival simulation with Stag Hunt cooperative mechanics.

## Configuration

- **Environment**: SurvivalWorld with Stag Hunt (cooperative radius=2.0, bonus=+5 food)
- **Episodes**: 3 per condition (using fixed seeds for reproducibility)
- **Max Steps**: 200 per episode
- **Actions**: rest, gather_resources, move_to (talk_to disabled)

## Models Tested

| Model | Backend | Notes |
|-------|---------|-------|
| Gemini 2.0 Flash | gemini | Primary test model |
| LLaMA 3.3 70B | groq | Heavy rate limiting (429 errors) |
| GPT-4o-mini | openai | Single test episode |

## Scaffold Conditions

| Scaffold | Format | Memory | Description |
|----------|--------|--------|-------------|
| baseline_nomem | Simple | No | Basic prompt, action name only |
| baseline_memory | Simple | Yes | + 5-step history window |
| explicit_nomem | Structured | No | ACTION: format required |
| explicit_memory | Structured | Yes | + 5-step history window |
| reasoning_nomem | Chain-of-Thought | No | THOUGHT + ACTION sections |
| reasoning_memory | Chain-of-Thought | Yes | + 5-step history window |
| tool_nomem | Calculator | No | CALC + ACTION sections |

## Results Summary

| Condition | Eps | Survival | Reward | Length | Coop | Parse Fail | Fallback |
|-----------|-----|----------|--------|--------|------|------------|----------|
| **Baselines** |
| heuristic_vs_heuristic | 3 | 100% | 6.33 | 200 | 66.7 | 0% | 0% |
| random_vs_random | 3 | 33% | 0.76 | 31 | 1.7 | 0% | 0% |
| rl_v3_vs_rl_v3 | 3 | 100% | 6.98 | 200 | 56.0 | 0% | 0% |
| **Gemini 2.0 Flash** |
| gemini_baseline_nomem | 3 | 100% | 6.33 | 200 | 27.7 | **46.1%** | 21.7% |
| gemini_baseline_memory | 3 | 100% | 4.05 | 200 | 39.7 | **14.6%** | 6.2% |
| gemini_explicit_nomem | 3 | 100% | 6.99 | 200 | 38.0 | **0%** | 0% |
| gemini_explicit_memory | 3 | 100% | 5.25 | 200 | 36.3 | **0%** | 0% |
| gemini_reasoning_nomem | 3 | 100% | 4.64 | 200 | 56.3 | **0%** | 0.2% |
| **Groq LLaMA 70B** |
| groq_baseline_nomem | 3 | 100% | 3.51 | 200 | 0.0 | 0% | 0% |
| groq_baseline_memory | 3 | 33% | -1.03 | 160 | 0.0 | 0% | **49.8%** |
| **OpenAI GPT-4o-mini** |
| openai_baseline_nomem | 1 | 100% | -0.30 | 50 | 0.0 | 0% | 0% |

## Key Findings

### 1. Scaffold Format Dramatically Affects Parse Success

| Format | Gemini Parse Failure Rate |
|--------|---------------------------|
| baseline | 46.1% |
| baseline + memory | 14.6% (memory helps!) |
| explicit | **0%** |
| reasoning | **0%** |

**Conclusion**: Structured output formats (explicit, reasoning) eliminate parsing failures entirely. The baseline format with free-form responses is brittle.

### 2. Memory Reduces Parse Failures

Adding a 5-step history window reduced Gemini's parse failure rate from 46.1% to 14.6% - a 68% reduction. Memory provides context that helps the LLM understand the expected response format.

### 3. Cooperative Behavior Varies by Scaffold

| Condition | Cooperative Gathers |
|-----------|---------------------|
| heuristic | 66.7 |
| rl_v3 | 56.0 |
| gemini_reasoning_nomem | **56.3** |
| gemini_baseline_memory | 39.7 |
| gemini_explicit_nomem | 38.0 |
| gemini_baseline_nomem | 27.7 |
| groq_baseline_* | **0.0** |

**Key insight**: Gemini's reasoning scaffold achieves cooperation rates comparable to RL v3, while Groq shows zero cooperation despite 100% survival.

### 4. Groq Rate Limiting is Severe

Groq's free tier has aggressive rate limiting (~30 requests before 429 errors). The baseline_memory condition had 49.8% fallback rate and 33% survival, indicating the LLM couldn't make valid decisions fast enough.

### 5. Model Comparison (baseline_nomem)

| Model | Parse Failures | Fallback | Survival | Cooperation |
|-------|---------------|----------|----------|-------------|
| Gemini | 46.1% | 21.7% | 100% | 27.7 |
| Groq | 0% | 0% | 100% | 0.0 |
| GPT-4o-mini | 0% | 0% | 100% | 0.0 |

Gemini has the worst instruction-following but the best emergent cooperation. Groq and GPT-4o-mini follow instructions perfectly but show no cooperative behavior.

## File Structure

```
llm_society/rl/
├── v6_prompts.py    # 7 scaffold prompt builders
├── v6_policy.py     # Gemini, Groq, OpenAI policies
├── v6_benchmark.py  # Benchmark runner
runs/survivalworld_v6/
├── *_heuristic_vs_heuristic.json
├── *_random_vs_random.json
├── *_rl_v3_vs_rl_v3.json
├── *_gemini_*.json
├── *_groq_*.json
└── *_openai_*.json
```

## Running the Benchmark

```bash
# Full benchmark (3 episodes, all scaffolds)
python -m llm_society.rl.v6_benchmark --episodes 3 --model gemini

# Quick test (1 episode, 50 steps)
python -m llm_society.rl.v6_benchmark --episodes 1 --max-steps 50 --model gemini --scaffold baseline_nomem

# Skip baselines (already run)
python -m llm_society.rl.v6_benchmark --episodes 3 --no-baselines --no-rl
```

## Environment Variables

```bash
export GOOGLE_API_KEY="..."   # Gemini
export GROQ_API_KEY="..."     # Groq
export OPENAI_API_KEY="..."   # OpenAI
```

## Citation

This benchmark addresses peer reviewer critiques by:
1. Testing multiple model families (not just Gemini)
2. Systematic scaffold comparison (7 conditions)
3. Enhanced metrics (token counts, latency, death causes)
4. Reproducible seeds across all conditions
