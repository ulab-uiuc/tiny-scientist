# Tiny-Scientist

## Introduction

Welcome to **Tiny-Scientist**, Tiny-Scientist is an automated research system that simulates the full lifecycle of a research process-—from ideation to implementation, paper writing, and review. This cutting-edge platform equips researchers with:

1. *End-to-End Automation*: Seamlessly generates novel research ideas, codes experiments, writes full papers, and conducts reviews—all powered by state-of-the-art LLMs.
2. *Modular Intelligence Agents*: Includes dedicated modules—Thinker, Coder, Writer, and Reviewer—each specialized for a distinct phase of the research workflow.
3. *Customizable and Extensible*: Designed for flexibility, allowing integration with various search tools and paper templates.

🔸 **Core Capabilities**

🚀 **Primary Research Functions**
* 🧠 Idea Generation & Refinement: Automatically generates innovative research ideas through iterative reasoning and refinement.
* 💻 Experiment Design & Coding: Translates research ideas into executable code with minimal human input. Supports experiment setup, baseline comparison.
* ✍️ Paper Writing: Generates full-length academic papers with well-structured sections, integrated citation management, and support for multiple conference templates.
* 📝 Paper Review: Evaluates academic drafts using structured reviewing criteria inspired by top-tier conferences.


## Get started

### Install from pip

You can install `tiny-scientist` from `pypi` to use it as a package:

```bash
pip install tiny-scientist
```

### Install from scratch

Use a virtual environment, e.g. with anaconda3:

```bash
conda create -n tiny-scientist python=3.10
conda activate tiny-scientist
curl -sSL https://install.python-poetry.org | python3
export PATH="$HOME/.local/bin:$PATH"
```

### Config environment variables
Envrionment variables and database related configs are required to code successfully. The recommended way to set all the required values is to use the provided configuration template.

Step 1: Copy the defaulty configuration template
```bash
cp config.template.toml config.toml
```

Step 2: Fill in the Required Fields
```bash
nano config.toml
```

Step 3: Then fill in the necessary API keys and parameters. For example:
```bash
[core]
# S2 API Key for accessing scientific research data
s2_api_key = "your-semantic-scholar-api-key"
```
