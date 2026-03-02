from __future__ import annotations

import sys
from contextlib import redirect_stdout
from typing import Any, Dict, Optional

from fastmcp import FastMCP  # type: ignore

from tiny_scientist.tool_impls import (
    ArxivDailyWatchTool,
    BenchmarkSearchTool,
    ClaimVerifierTool,
    CodeSearchTool,
    DatasetSearchTool,
    NewsSearchTool,
    PaperSearchTool,
    PatentSearchTool,
    RepoRuntimeProbeTool,
    ScholarGraphSearchTool,
    TableExtractorTool,
    WebSearchTool,
)

app = FastMCP("tiny-scientist-research", description="Research tools MCP server")


@app.tool(name="paper_search", description="Search academic papers")
def run_paper_search(query: str, result_limit: int = 10) -> Dict[str, Dict[str, str]]:
    with redirect_stdout(sys.stderr):
        return PaperSearchTool().forward(query=query, result_limit=result_limit)


@app.tool(name="code_search", description="Search local or GitHub code")
def run_code_search(
    query: str,
    search_type: str = "local",
    result_limit: int = 10,
) -> Dict[str, Dict[str, str]]:
    with redirect_stdout(sys.stderr):
        return CodeSearchTool().forward(
            query=query,
            search_type=search_type,
            result_limit=result_limit,
        )


@app.tool(name="web_search", description="Search the public web")
def run_web_search(query: str, result_limit: int = 5) -> Dict[str, Dict[str, str]]:
    with redirect_stdout(sys.stderr):
        return WebSearchTool().forward(query=query, result_limit=result_limit)


@app.tool(
    name="scholar_graph_search",
    description="Traverse scholarly citation or reference graph",
)
def run_scholar_graph_search(
    query: str,
    mode: str = "citations",
    result_limit: int = 10,
) -> Dict[str, Dict[str, str]]:
    with redirect_stdout(sys.stderr):
        return ScholarGraphSearchTool().forward(
            query=query,
            mode=mode,
            result_limit=result_limit,
        )


@app.tool(name="patent_search", description="Search patent records")
def run_patent_search(query: str, result_limit: int = 10) -> Dict[str, Dict[str, str]]:
    with redirect_stdout(sys.stderr):
        return PatentSearchTool().forward(query=query, result_limit=result_limit)


@app.tool(name="dataset_search", description="Search machine learning datasets")
def run_dataset_search(query: str, result_limit: int = 10) -> Dict[str, Dict[str, str]]:
    with redirect_stdout(sys.stderr):
        return DatasetSearchTool().forward(query=query, result_limit=result_limit)


@app.tool(name="benchmark_search", description="Search benchmark tasks and leaderboards")
def run_benchmark_search(
    query: str, result_limit: int = 10
) -> Dict[str, Dict[str, str]]:
    with redirect_stdout(sys.stderr):
        return BenchmarkSearchTool().forward(query=query, result_limit=result_limit)


@app.tool(name="arxiv_daily_watch", description="Search recent arXiv papers")
def run_arxiv_daily_watch(
    query: str,
    days: int = 7,
    result_limit: int = 10,
) -> Dict[str, Dict[str, str]]:
    with redirect_stdout(sys.stderr):
        return ArxivDailyWatchTool().forward(
            query=query,
            days=days,
            result_limit=result_limit,
        )


@app.tool(name="news_search", description="Search recent news")
def run_news_search(query: str, result_limit: int = 10) -> Dict[str, Dict[str, str]]:
    with redirect_stdout(sys.stderr):
        return NewsSearchTool().forward(query=query, result_limit=result_limit)


@app.tool(
    name="repo_runtime_probe",
    description="Inspect repository runtime metadata and likely entrypoints",
)
def run_repo_runtime_probe(repo_path: str) -> Dict[str, Any]:
    with redirect_stdout(sys.stderr):
        return RepoRuntimeProbeTool().forward(repo_path=repo_path)


@app.tool(name="table_extractor", description="Extract table-like blocks from a PDF")
def run_table_extractor(pdf_path: str, max_tables: int = 5) -> Dict[str, Any]:
    with redirect_stdout(sys.stderr):
        return TableExtractorTool().forward(pdf_path=pdf_path, max_tables=max_tables)


@app.tool(name="claim_verifier", description="Collect candidate evidence for claims")
def run_claim_verifier(claims_json: str, per_claim_limit: int = 5) -> Dict[str, Any]:
    with redirect_stdout(sys.stderr):
        return ClaimVerifierTool().forward(
            claims_json=claims_json,
            per_claim_limit=per_claim_limit,
        )


if __name__ == "__main__":
    app.run()
