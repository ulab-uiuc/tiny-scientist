from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import toml

from tiny_scientist.budget_checker import BudgetChecker
from tiny_scientist.utils.mcp_client import call_mcp_tool

CONFIG_PATH = Path(__file__).resolve().parent / "config.toml"
config = toml.load(CONFIG_PATH) if CONFIG_PATH.exists() else {"core": {}}


class BaseTool:
    def __init__(self, cost_tracker: Optional[BudgetChecker] = None) -> None:
        self.cost_tracker = cost_tracker or BudgetChecker()
        self.github_token = config.get("core", {}).get("github_token")

    def run(self, *args: Any, **kwargs: Any) -> Dict[str, Dict[str, str]]:
        raise NotImplementedError


class CodeSearchTool(BaseTool):
    SERVER = "code_search"
    TOOL_NAME = "code_search.run"

    def run(
        self, query: str, search_type: str = "repositories"
    ) -> Dict[str, Dict[str, str]]:
        payload = {"query": query, "search_type": search_type}
        result = call_mcp_tool(self.SERVER, self.TOOL_NAME, payload)
        self.cost_tracker.report()
        return result or {}


class PaperSearchTool(BaseTool):
    SERVER = "paper_search"
    TOOL_NAME = "paper_search.run"

    def __init__(
        self,
        s2_api_key: Optional[str] = None,
        engine: Optional[str] = None,
        disable_fallback: bool = False,
        cost_tracker: Optional[BudgetChecker] = None,
    ) -> None:
        super().__init__(cost_tracker=cost_tracker)
        raw_key = (
            s2_api_key
            or os.environ.get("S2_API_KEY")
            or config.get("core", {}).get("s2_api_key")
        )
        self.s2_api_key = raw_key.strip() if isinstance(raw_key, str) else raw_key
        self.disable_fallback = disable_fallback
        configured_engine = config.get("core", {}).get("engine")
        if engine:
            self.engine = engine
        elif configured_engine:
            self.engine = configured_engine
        elif self.s2_api_key:
            self.engine = "semanticscholar"
        else:
            self.engine = "openalex"

    def run(self, query: str, result_limit: int = 10) -> Dict[str, Dict[str, str]]:
        payload = {
            "query": query,
            "result_limit": result_limit,
            "s2_api_key": self.s2_api_key,
            "engine": self.engine,
            "disable_fallback": self.disable_fallback,
        }
        result = call_mcp_tool(self.SERVER, self.TOOL_NAME, payload)
        self.cost_tracker.report()
        return result or {}


class DrawerTool(BaseTool):
    SERVER = "drawer"
    TOOL_NAME = "drawer.run"

    def __init__(
        self,
        model: str,
        prompt_template_dir: Optional[str] = None,
        temperature: float = 0.75,
        cost_tracker: Optional[BudgetChecker] = None,
    ) -> None:
        super().__init__(cost_tracker=cost_tracker)
        self.model = model
        self.prompt_template_dir = prompt_template_dir
        self.temperature = temperature

    def run(self, query: str) -> Dict[str, Dict[str, str]]:
        payload = {
            "query": query,
            "model": self.model,
            "prompt_template_dir": self.prompt_template_dir,
            "temperature": self.temperature,
        }
        result = call_mcp_tool(self.SERVER, self.TOOL_NAME, payload)
        self.cost_tracker.report()
        return result or {}


class DockerExperimentRunner:
    SERVER = "docker_runner"
    STATUS_TOOL = "docker_runner.status"
    RUN_TOOL = "docker_runner.run_experiment"
    CLEANUP_TOOL = "docker_runner.cleanup"

    @staticmethod
    def extract_missing_package(stderr: str) -> str:
        """Extract missing package name from stderr with package name mapping."""

        package_mapping = {
            "sklearn": "scikit-learn",
            "cv2": "opencv-python",
            "PIL": "pillow",
            "yaml": "pyyaml",
            "bs4": "beautifulsoup4",
            "lxml": "lxml",
            "nltk": "nltk",
            "spacy": "spacy",
            "gensim": "gensim",
            "wordcloud": "wordcloud",
            "plotly": "plotly",
            "bokeh": "bokeh",
            "dash": "dash",
            "streamlit": "streamlit",
            "fastapi": "fastapi",
            "uvicorn": "uvicorn",
            "sqlalchemy": "sqlalchemy",
            "psycopg2": "psycopg2-binary",
            "pymongo": "pymongo",
            "redis": "redis",
            "celery": "celery",
            "flask": "flask",
            "django": "django",
            "boto3": "boto3",
            "google": "google-api-python-client",
            "googleapiclient": "google-api-python-client",
            "grpc": "grpcio",
            "tensorflow": "tensorflow",
            "torch": "torch",
            "torchvision": "torchvision",
            "torchaudio": "torchaudio",
            "keras": "keras",
            "jax": "jax",
            "jaxlib": "jaxlib",
            "xgboost": "xgboost",
            "lightgbm": "lightgbm",
            "catboost": "catboost",
            "prophet": "prophet",
            "pymc3": "pymc3",
            "pystan": "pystan",
            "fbprophet": "fbprophet",
            "transformers": "transformers",
            "datasets": "datasets",
            "evaluate": "evaluate",
            "sentencepiece": "sentencepiece",
            "langchain": "langchain",
            "accelerate": "accelerate",
            "bitsandbytes": "bitsandbytes",
            "triton": "triton",
            "diffusers": "diffusers",
            "opencv": "opencv-python",
            "matplotlib": "matplotlib",
            "seaborn": "seaborn",
            "scipy": "scipy",
            "sympy": "sympy",
            "statsmodels": "statsmodels",
            "pandas": "pandas",
            "numpy": "numpy",
            "polars": "polars",
            "pyarrow": "pyarrow",
            "fastparquet": "fastparquet",
            "openpyxl": "openpyxl",
            "xlrd": "xlrd",
            "xlwt": "xlwt",
            "odf": "odfpy",
            "tabula": "tabula-py",
            "pdfplumber": "pdfplumber",
            "pymupdf": "pymupdf",
            "pypdf": "pypdf",
            "reportlab": "reportlab",
            "weasyprint": "weasyprint",
            "jinja2": "jinja2",
            "markdown": "markdown",
            "rst": "docutils",
            "sphinx": "sphinx",
            "mkdocs": "mkdocs",
            "jupyter": "jupyter",
            "ipython": "ipython",
            "notebook": "notebook",
            "jupyterlab": "jupyterlab",
            "voila": "voila",
            "nbconvert": "nbconvert",
            "papermill": "papermill",
            "nbdime": "nbdime",
            "nbstripout": "nbstripout",
            "jupytext": "jupytext",
        }

        for line in stderr.splitlines():
            if "ModuleNotFoundError" in line:
                parts = line.split("'")
                if len(parts) >= 2:
                    import_name = parts[1]
                    return package_mapping.get(import_name, import_name)
        return "unknown-package"

    def __init__(
        self,
        docker_image: str = "tiny-scientist-ml",
        docker_base: str = "python:3.11-slim",
    ) -> None:
        self.docker_image = docker_image
        self.docker_base = docker_base
        status = call_mcp_tool(
            self.SERVER,
            self.STATUS_TOOL,
            {
                "docker_image": self.docker_image,
                "docker_base": self.docker_base,
            },
        )
        if isinstance(status, dict):
            self.use_docker = bool(status.get("use_docker", True))
        else:
            self.use_docker = True

    def run_experiment_in_docker(
        self,
        experiment_code: str,
        run_num: int,
        output_dir: str,
        timeout: int = 7200,
    ) -> Optional[Any]:
        payload = {
            "experiment_code": experiment_code,
            "run_num": run_num,
            "output_dir": output_dir,
            "timeout": timeout,
            "docker_image": self.docker_image,
            "docker_base": self.docker_base,
        }
        result = call_mcp_tool(self.SERVER, self.RUN_TOOL, payload)
        if isinstance(result, dict) and "status_code" in result:
            return result["status_code"], result.get("logs", "")
        return result

    def cleanup_docker_images(self) -> None:
        call_mcp_tool(
            self.SERVER,
            self.CLEANUP_TOOL,
            {"docker_image": self.docker_image, "docker_base": self.docker_base},
        )
