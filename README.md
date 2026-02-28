<div style="width: 100%;">
  <img src="assets/tiny_scientist.png" style="width: 100%;"></img>
</div>

<h1 align="center">TinyScientist: A Lightweight Framework for Building Research Agents</h1>

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/tiny-scientist)](https://pypi.org/project/tiny-scientist/)
[![Python 3.10](https://img.shields.io/badge/python-%E2%89%A53.10-blue)](https://www.python.org/downloads/release/python-3109/)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-red)](https://github.com/hiyouga/LLaMA-Factory/pulls)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![bear-ified](https://raw.githubusercontent.com/beartype/beartype-assets/main/badge/bear-ified.svg)](https://beartype.readthedocs.io)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![arXiv](https://img.shields.io/badge/arXiv-2510.06579-b31b1b.svg)](https://arxiv.org/abs/2510.06579)

</div>

# News

- **Oct 11, 2025** — Our paper is out on arXiv ([2510.06579](https://arxiv.org/abs/2510.06579)) and accepted to EMNLP 2025 (Demo Track) 🎉

# Introduction

**Tiny-Scientist** is a lightweight, user-friendly framework for automating the entire lifecycle of scientific research—**from ideation to implementation, writing, and review**. Designed for flexibility, it integrates smoothly with your favorite LLMs and search tools.

#### Core Features

- 🧠 **Think**: Generate structured research ideas from an intent string.
- 💻 **Code**: Automatically generate and run experiments based on the idea.
- ✍️ **Write**: Convert your results and ideas into a conference-style paper.
- 📝 **Review**: Review any form of paper and output structured feedback in JSON.
- 🔧 **MCP**: The extensible tool use protocol by Anthropic

#### Software Architecture

Our codebase is structured around three core components to support an extensible framework: **core**, **tools**, and **formatters**. The **core** module provides essential functionalities, **tools** enhance and extend these core capabilities, and **formatters** handle input/output tasks such as LaTeX template rendering.

<p align="center">
  <img src="assets/architecture.png" alt="architecture" width="100%"/>
</p>


# Installation

#### Option 1: Install via pip (recommended)

```bash
pip install tiny-scientist
```

#### Option 2: Install from source

```bash
# create conda environment
conda create -n tiny-scientist python=3.10
conda activate tiny-scientist

# Install Poetry
curl -sSL https://install.python-poetry.org | python3
export PATH="$HOME/.local/bin:$PATH"

# Install dependencies
poetry install
```

# Get started

TinyScientist now uses Claude Agent SDK-backed stages by default (`think`, `code`, `write`, `review`). OpenAI Agents SDK remains available as an explicit runtime choice.

#### 1) Required runtime setup

Set your model API key:

```bash
export OPENAI_API_KEY=your-key-here
# or DEEPSEEK_API_KEY / ANTHROPIC_API_KEY depending on your model
```

If you installed from source and see `ModuleNotFoundError: No module named 'claude_agent_sdk'`, install:

```bash
pip install claude-agent-sdk
```

If you want the OpenAI runtime as well, install:

```bash
pip install openai-agents
```

`smolagents` is no longer required for the main TinyScientist runtime. It is only needed for legacy compatibility helpers.

#### 2) Tool providers (strict mode)

Tooling runs in strict provider mode (no automatic fallback). If a selected provider is unavailable, the tool call fails.

Core switches:

```bash
# Web search provider: duckduckgo | tavily | serpapi | brave
export WEB_SEARCH_PROVIDER=duckduckgo

# Diagram backend: llm_svg | nano-banana
export DRAWER_BACKEND=llm_svg
```

Optional provider keys:

```bash
export S2_API_KEY=...             # semantic scholar tools
export NEWSAPI_KEY=...            # news_search
export TAVILY_API_KEY=...         # if WEB_SEARCH_PROVIDER=tavily
export SERPAPI_API_KEY=...        # if WEB_SEARCH_PROVIDER=serpapi
export BRAVE_SEARCH_API_KEY=...   # if WEB_SEARCH_PROVIDER=brave

# nano-banana image generation (optional)
export NANO_BANANA_MODEL=gpt-image-1
```

#### 3) Minimal Python usage (unchanged API)

The minimal Python API is still the same:

```python
from tiny_scientist import TinyScientist

scientist = TinyScientist(model="claude-3-5-sonnet-20241022", budget=1.0)

idea = scientist.think(
    intent="Benchmarking adaptive step size strategies using a convex quadratic optimization function"
)
status, experiment_dir = scientist.code(idea=idea)

if status:
    pdf_path = scientist.write(idea=idea, experiment_dir=experiment_dir)
    review = scientist.review(pdf_path=pdf_path)
```

Use `agent_sdk` to select the runtime explicitly when you want the OpenAI backend:

```python
scientist = TinyScientist(model="gpt-4o", agent_sdk="openai")
```

Supported values today are `claude` and `openai`. The default is `claude`.

#### 3.1) Skills and MCP

TinyScientist now supports project skills from both:

- `.claude/skills`
- `.agents/skills`

Claude backend:

- Uses Claude's native filesystem skill loading via `.claude/skills`
- Loads Claude skills from both user and project settings sources
- Does not inject `SKILL.md` contents into prompts; skills are consumed only through Claude's native `Skill` tool flow
- Uses generated `.tiny_scientist.generated.mcp.json` to mount TinyScientist MCP research tools

OpenAI backend:

- Continues to inject local `SKILL.md` content into agent instructions
- Also supports official OpenAI shell-mounted skills when you provide skill specs as JSON:

```bash
export OPENAI_AGENT_SKILLS_JSON='[{"type":"skill_reference","skill_id":"skill_xxx","version":"1"}]'
```

You can also scope mounted OpenAI skills per stage:

```bash
export OPENAI_AGENT_SKILLS_THINKER_JSON='[...]'
export OPENAI_AGENT_SKILLS_CODER_JSON='[...]'
export OPENAI_AGENT_SKILLS_WRITER_JSON='[...]'
export OPENAI_AGENT_SKILLS_REVIEWER_JSON='[...]'
```

#### 4) Non-OpenAI-compatible endpoints (advanced)

For OpenAI-compatible gateways:

```bash
export OPENAI_API_BASE=http://your-endpoint/v1
export OPENAI_API_KEY=your-key-here
```

Then use models like `openai/qwen3-30b-a3b`.

#### 5) Built-in research tools

Agents can use these built-in tools during thinking/coding/writing/review:

- `web_search`
- `paper_search`
- `scholar_graph_search`
- `benchmark_search`
- `dataset_search`
- `code_search`
- `repo_runtime_probe`
- `arxiv_daily_watch`
- `news_search`
- `patent_search`
- `table_extractor`
- `claim_verifier`
- `generate_diagram` (writer path)

# Managing API Keys (Optional)

You can configure keys using a `.toml` file for convenience beyond exporting.

#### Step 1: Copy the template

```bash
cp config.template.toml config.toml
```

#### Step 2: Fill in your API credentials

Edit `config.toml` to include your keys, such as:

```toml
[core]
llm_api_key = "xxxx"
```

No need to export environment variables manually—just set this once.

# Developing

#### Develop Demo
To develop a demo (Both frontend and backend):
```bash
python backend/app.py
```
```bash
cd frontend
npm install
npm start
```
# Q&A

If you face "cairo"-related errors, cario is a system-level dependency, please run `conda install -c conda-forge cairo` or `brew install cairo`.

If you face errors related to pdflatex, this is also a system-level dependency for latex rendering, please run `brew install --cask mactex`.

# Contribution

We’re working on extending support for more tools, models, and paper formats. Contributions welcome!

# Citation

```
@misc{tinyscientist,
author       = {Haofei Yu and Keyang Xuan and Fenghai Li and Kunlun Zhu and Zijie Lei and Jiaxun Zhang and Ziheng Qi and Jiaxuan You},
title        = {TinyScientist: A Lightweight Framework for Building Research Agents},
howpublished = {https://github.com/ulab-uiuc/tiny-scientist},
note         = {Accessed: 2025-04-14},
year         = {2025}
}
```
