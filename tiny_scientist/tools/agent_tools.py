"""OpenAI Agents SDK @function_tool wrappers for smolagents tool instances."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, List

from agents import function_tool

if TYPE_CHECKING:
    from tiny_scientist.smolagents_tools import (
        CodeSearchTool,
        DrawerTool,
        PaperSearchTool,
        ScholarGraphSearchTool,
        PatentSearchTool,
        DatasetSearchTool,
        BenchmarkSearchTool,
        ArxivDailyWatchTool,
        NewsSearchTool,
        RepoRuntimeProbeTool,
        TableExtractorTool,
        ClaimVerifierTool,
        WebSearchTool,
        DockerExperimentRunner,
        ReadFileTool,
        RunExperimentTool,
        WriteFileTool,
    )


def make_write_file_tool(tool_instance: "WriteFileTool"):
    """Wrap a WriteFileTool instance as an openai-agents function_tool."""

    @function_tool
    def write_file(filename: str, content: str) -> str:
        """Write content to a file in the experiment directory.

        Use this to create or update experiment.py with complete Python code.
        """
        return tool_instance.forward(filename, content)

    return write_file


def make_read_file_tool(tool_instance: "ReadFileTool"):
    """Wrap a ReadFileTool instance as an openai-agents function_tool."""

    @function_tool
    def read_file(filename: str) -> str:
        """Read content from a file in the experiment directory.

        Use this to inspect the current content of experiment.py.
        """
        return tool_instance.forward(filename)

    return read_file


def make_run_experiment_tool(tool_instance: "RunExperimentTool"):
    """Wrap a RunExperimentTool instance as an openai-agents function_tool."""

    @function_tool
    def run_experiment(run_num: int) -> str:
        """Run the experiment.py script in Docker or locally and return the result.

        After writing experiment.py, call this to execute it.
        """
        return tool_instance.forward(run_num)

    return run_experiment


def make_paper_search_tool(tool_instance: "PaperSearchTool"):
    """Wrap a PaperSearchTool instance as an openai-agents function_tool."""

    @function_tool
    def paper_search(query: str, result_limit: int = 10) -> str:
        """Search academic papers by query and return structured metadata as JSON text."""
        return json.dumps(
            tool_instance.forward(query=query, result_limit=result_limit),
            ensure_ascii=False,
        )

    return paper_search


def make_code_search_tool(tool_instance: "CodeSearchTool"):
    """Wrap a CodeSearchTool instance as an openai-agents function_tool."""

    @function_tool
    def code_search(
        query: str, search_type: str = "local", result_limit: int = 10
    ) -> str:
        """Search GitHub repositories or code snippets and return JSON text."""
        return json.dumps(
            tool_instance.forward(
                query=query, search_type=search_type, result_limit=result_limit
            ),
            ensure_ascii=False,
        )

    return code_search


def make_drawer_tool(tool_instance: "DrawerTool"):
    """Wrap a DrawerTool instance as an openai-agents function_tool."""

    @function_tool
    def generate_diagram(section_name: str, section_content: str) -> str:
        """Generate a diagram (SVG + summary) for a paper section and return JSON text."""
        payload = json.dumps(
            {"section_name": section_name, "section_content": section_content}
        )
        return json.dumps(tool_instance.forward(payload), ensure_ascii=False)

    return generate_diagram


def make_web_search_tool(tool_instance: "WebSearchTool"):
    """Wrap a WebSearchTool instance as an openai-agents function_tool."""

    @function_tool
    def web_search(query: str, result_limit: int = 5) -> str:
        """Search the web for up-to-date context and return JSON text."""
        return json.dumps(
            tool_instance.forward(query=query, result_limit=result_limit),
            ensure_ascii=False,
        )

    return web_search


def make_scholar_graph_search_tool(tool_instance: "ScholarGraphSearchTool"):
    @function_tool
    def scholar_graph_search(
        query: str, mode: str = "citations", result_limit: int = 10
    ) -> str:
        """Traverse citation graph for related scholarly papers."""
        return json.dumps(
            tool_instance.forward(query=query, mode=mode, result_limit=result_limit),
            ensure_ascii=False,
        )

    return scholar_graph_search


def make_patent_search_tool(tool_instance: "PatentSearchTool"):
    @function_tool
    def patent_search(query: str, result_limit: int = 10) -> str:
        """Search patent records for novelty/prior-art analysis."""
        return json.dumps(
            tool_instance.forward(query=query, result_limit=result_limit),
            ensure_ascii=False,
        )

    return patent_search


def make_dataset_search_tool(tool_instance: "DatasetSearchTool"):
    @function_tool
    def dataset_search(query: str, result_limit: int = 10) -> str:
        """Search machine learning datasets relevant to the task."""
        return json.dumps(
            tool_instance.forward(query=query, result_limit=result_limit),
            ensure_ascii=False,
        )

    return dataset_search


def make_benchmark_search_tool(tool_instance: "BenchmarkSearchTool"):
    @function_tool
    def benchmark_search(query: str, result_limit: int = 10) -> str:
        """Search benchmark tasks and leaderboard-like references."""
        return json.dumps(
            tool_instance.forward(query=query, result_limit=result_limit),
            ensure_ascii=False,
        )

    return benchmark_search


def make_arxiv_daily_watch_tool(tool_instance: "ArxivDailyWatchTool"):
    @function_tool
    def arxiv_daily_watch(query: str, days: int = 7, result_limit: int = 10) -> str:
        """Find recently submitted arXiv papers for a topic."""
        return json.dumps(
            tool_instance.forward(query=query, days=days, result_limit=result_limit),
            ensure_ascii=False,
        )

    return arxiv_daily_watch


def make_news_search_tool(tool_instance: "NewsSearchTool"):
    @function_tool
    def news_search(query: str, result_limit: int = 10) -> str:
        """Search recent news and announcements relevant to the topic."""
        return json.dumps(
            tool_instance.forward(query=query, result_limit=result_limit),
            ensure_ascii=False,
        )

    return news_search


def make_repo_runtime_probe_tool(tool_instance: "RepoRuntimeProbeTool"):
    @function_tool
    def repo_runtime_probe(repo_path: str) -> str:
        """Inspect repository runtime metadata and likely entrypoints."""
        return json.dumps(tool_instance.forward(repo_path=repo_path), ensure_ascii=False)

    return repo_runtime_probe


def make_table_extractor_tool(tool_instance: "TableExtractorTool"):
    @function_tool
    def table_extractor(pdf_path: str, max_tables: int = 5) -> str:
        """Extract table-like blocks from PDF text."""
        return json.dumps(
            tool_instance.forward(pdf_path=pdf_path, max_tables=max_tables),
            ensure_ascii=False,
        )

    return table_extractor


def make_claim_verifier_tool(tool_instance: "ClaimVerifierTool"):
    @function_tool
    def claim_verifier(claims_json: str, per_claim_limit: int = 5) -> str:
        """Collect web and paper evidence candidates for claims."""
        return json.dumps(
            tool_instance.forward(
                claims_json=claims_json, per_claim_limit=per_claim_limit
            ),
            ensure_ascii=False,
        )

    return claim_verifier


def build_research_tools(
    model: str, include_drawer: bool = False, include_extended: bool = True
) -> List[Any]:
    """Build a consistent set of research tools for OpenAI agents.

    Always includes:
    - paper_search
    - code_search
    - web_search

    Optionally includes:
    - generate_diagram (when include_drawer=True)
    """
    from tiny_scientist.smolagents_tools import (
        ArxivDailyWatchTool,
        BenchmarkSearchTool,
        ClaimVerifierTool,
        CodeSearchTool,
        DatasetSearchTool,
        DrawerTool,
        NewsSearchTool,
        PatentSearchTool,
        PaperSearchTool,
        RepoRuntimeProbeTool,
        ScholarGraphSearchTool,
        TableExtractorTool,
        WebSearchTool,
    )

    tools: List[Any] = [
        make_paper_search_tool(PaperSearchTool()),
        make_code_search_tool(CodeSearchTool()),
        make_web_search_tool(WebSearchTool()),
    ]
    if include_extended:
        tools.extend(
            [
                make_scholar_graph_search_tool(ScholarGraphSearchTool()),
                make_patent_search_tool(PatentSearchTool()),
                make_dataset_search_tool(DatasetSearchTool()),
                make_benchmark_search_tool(BenchmarkSearchTool()),
                make_arxiv_daily_watch_tool(ArxivDailyWatchTool()),
                make_news_search_tool(NewsSearchTool()),
                make_repo_runtime_probe_tool(RepoRuntimeProbeTool()),
                make_table_extractor_tool(TableExtractorTool()),
                make_claim_verifier_tool(ClaimVerifierTool()),
            ]
        )

    if include_drawer:
        tools.append(make_drawer_tool(DrawerTool(model=model)))

    return tools
