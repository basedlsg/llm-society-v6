# LLM Society - 2,500-Agent LLM-Driven Society Simulation

> Large-scale multi-agent society simulation powered by LLMs with economic, social, and spatial dynamics

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Mesa](https://img.shields.io/badge/Mesa-1.0+-green.svg)](https://mesa.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

LLM Society is an ambitious agent-based modeling system that simulates complex societal dynamics with up to 2,500 intelligent agents. Each agent is powered by LLMs (Gemini, GPT-4, Claude) for decision-making, social interactions, economic transactions, and spatial reasoning. The simulation includes economic systems, social networks, family dynamics, and cultural evolution.

### Key Features

- **Large-Scale Simulation**: 50-2,500 agents with LLM-driven behavior
- **Multi-LLM Integration**: Google Gemini, OpenAI GPT-4, Anthropic Claude
- **Economic System**: Dynamic market economy with supply/demand
- **Social Dynamics**: Relationships, families, cultural transmission
- **Spatial Reasoning**: Location-based interactions and movement
- **FlameGPU Integration**: GPU-accelerated simulation (in progress)
- **Real-Time Dashboard**: Monitor society metrics live
- **Experiment Framework**: Reproducible simulation experiments

## Quick Start

```bash
# Clone and install
git clone https://github.com/basedlsg/llm-society.git
cd llm-society
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with API keys

# Run basic simulation
python -m llm_society.main --agents 50 --steps 100
```

## Architecture

```
llm-society/
├── llm_society/
│   ├── agents/          # LLM-powered agents
│   ├── economics/       # Economic system
│   ├── social/          # Social dynamics
│   ├── simulation/      # Simulation engine
│   ├── flame_gpu/       # GPU acceleration
│   ├── database/        # State persistence
│   └── monitoring/      # Metrics & dashboard
├── tests/               # Demos and tests
└── docs/                # Documentation
```

## Known Issues ⚠️

See [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) for 186-hour production roadmap.

**Critical Bugs:**
1. main.py has broken imports (lines 100, 119) - files don't exist
2. llm_agent.py has indentation errors (lines 812-869)
3. FlameGPU integration incomplete
4. Zero test coverage

## Documentation

- [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) - 186-hour production roadmap
- [LLM_SOCIETY_README.md](docs/LLM_SOCIETY_README.md) - Detailed documentation

## Dependencies

Core: Mesa, Google Generative AI, OpenAI, Anthropic, NumPy, NetworkX

See [requirements.txt](requirements.txt) for complete list.

---

**Status**: Research Prototype (60% complete)
**Version**: 1.0.0
**Maintainer**: Based LSG
**Separated from**: [NOUS monorepo](https://github.com/basedlsg/NOUS) (2025-11-15)
