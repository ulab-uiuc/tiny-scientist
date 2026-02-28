from __future__ import annotations

import importlib
import io
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_fake_agents() -> type:
    fake_agents = types.ModuleType("agents")

    class FakeAgent:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class FakeRunner:
        @staticmethod
        def run_sync(agent, prompt):
            return types.SimpleNamespace(final_output="", raw_responses=[])

    class FakeShellTool:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    def function_tool(fn):
        return fn

    def set_default_openai_client(_client):
        return None

    fake_agents.Agent = FakeAgent
    fake_agents.Runner = FakeRunner
    fake_agents.ShellTool = FakeShellTool
    fake_agents.function_tool = function_tool
    fake_agents.set_default_openai_client = set_default_openai_client
    sys.modules["agents"] = fake_agents
    return FakeShellTool


def _install_fake_runtime_modules() -> None:
    import importlib.resources as importlib_resources

    fake_fitz = types.ModuleType("fitz")
    fake_smolagents = types.ModuleType("smolagents")
    fake_google = types.ModuleType("google")
    fake_google_generativeai = types.ModuleType("google.generativeai")
    fake_google_generativeai_types = types.ModuleType("google.generativeai.types")
    fake_pymupdf = types.ModuleType("pymupdf")
    fake_pymupdf4llm = types.ModuleType("pymupdf4llm")
    fake_pypdf = types.ModuleType("pypdf")
    fake_reportlab = types.ModuleType("reportlab")
    fake_reportlab_lib = types.ModuleType("reportlab.lib")
    fake_reportlab_lib_colors = types.ModuleType("reportlab.lib.colors")
    fake_reportlab_pdfgen = types.ModuleType("reportlab.pdfgen")
    fake_reportlab_pdfgen_canvas = types.ModuleType("reportlab.pdfgen.canvas")

    class _FakePage:
        def get_text(self, *_args, **_kwargs):
            return ""

    class _FakeDoc(list):
        def __init__(self):
            super().__init__([_FakePage()])

    def open_doc(*_args, **_kwargs):
        return _FakeDoc()

    class _FakeTool:
        def __init__(self, *args, **kwargs):
            pass

    class _FakeGenerationConfig:
        def __init__(self, *args, **kwargs):
            pass

    class _FakePdfReader:
        def __init__(self, *args, **kwargs):
            self.pages = []

    class _FakePdfWriter:
        def __init__(self, *args, **kwargs):
            pass

    class _FakeColor:
        def __init__(self, *args, **kwargs):
            pass

    class _FakeCanvas:
        def __init__(self, *args, **kwargs):
            pass

        def setFont(self, *args, **kwargs):
            pass

        def setFillColor(self, *args, **kwargs):
            pass

        def saveState(self):
            pass

        def translate(self, *args, **kwargs):
            pass

        def rotate(self, *args, **kwargs):
            pass

        def drawCentredString(self, *args, **kwargs):
            pass

        def restoreState(self):
            pass

        def save(self):
            pass

    fake_fitz.open = open_doc
    fake_smolagents.Tool = _FakeTool
    fake_google_generativeai_types.GenerationConfig = _FakeGenerationConfig
    fake_google_generativeai.types = fake_google_generativeai_types
    fake_google.generativeai = fake_google_generativeai
    fake_pypdf.PdfReader = _FakePdfReader
    fake_pypdf.PdfWriter = _FakePdfWriter
    fake_reportlab_lib_colors.Color = _FakeColor
    fake_reportlab_pdfgen_canvas.Canvas = _FakeCanvas
    fake_reportlab.lib = fake_reportlab_lib
    fake_reportlab.pdfgen = fake_reportlab_pdfgen
    fake_reportlab_lib.colors = fake_reportlab_lib_colors
    fake_reportlab_pdfgen.canvas = fake_reportlab_pdfgen_canvas
    if not hasattr(importlib_resources, "files"):
        class _FakeResourcePath:
            def joinpath(self, *_args, **_kwargs):
                return self

            def open(self, *_args, **_kwargs):
                return io.BytesIO(b"")

        importlib_resources.files = lambda _package: _FakeResourcePath()
    sys.modules["fitz"] = fake_fitz
    sys.modules["smolagents"] = fake_smolagents
    sys.modules["google"] = fake_google
    sys.modules["google.generativeai"] = fake_google_generativeai
    sys.modules["google.generativeai.types"] = fake_google_generativeai_types
    sys.modules["pymupdf"] = fake_pymupdf
    sys.modules["pymupdf4llm"] = fake_pymupdf4llm
    sys.modules["pypdf"] = fake_pypdf
    sys.modules["reportlab"] = fake_reportlab
    sys.modules["reportlab.lib"] = fake_reportlab_lib
    sys.modules["reportlab.lib.colors"] = fake_reportlab_lib_colors
    sys.modules["reportlab.pdfgen"] = fake_reportlab_pdfgen
    sys.modules["reportlab.pdfgen.canvas"] = fake_reportlab_pdfgen_canvas


def _clear_tiny_scientist_modules() -> None:
    for name in list(sys.modules.keys()):
        if name == "tiny_scientist" or name.startswith("tiny_scientist."):
            del sys.modules[name]


def test_openai_sdk_smoke_mounts_shell_skills(monkeypatch, tmp_path: Path) -> None:
    fake_shell_tool = _install_fake_agents()
    _install_fake_runtime_modules()
    _clear_tiny_scientist_modules()

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv(
        "OPENAI_AGENT_SKILLS_JSON",
        '[{"type":"skill_reference","skill_id":"skill_test","version":"1"}]',
    )

    thinker_module = importlib.import_module("tiny_scientist.thinker")
    writer_module = importlib.import_module("tiny_scientist.writer")
    reviewer_module = importlib.import_module("tiny_scientist.reviewer")
    coder_module = importlib.import_module("tiny_scientist.coder")

    thinker = thinker_module.Thinker(
        model="gpt-4o",
        output_dir=str(tmp_path),
        tools=[],
        agent_sdk="openai",
    )
    writer = writer_module.Writer(
        model="gpt-4o",
        output_dir=str(tmp_path),
        template="acl",
        agent_sdk="openai",
    )
    reviewer = reviewer_module.Reviewer(
        model="gpt-4o",
        tools=[],
        agent_sdk="openai",
    )
    coder = coder_module.Coder(
        model="gpt-4o",
        output_dir=str(tmp_path),
        use_docker=False,
        agent_sdk="openai",
    )
    coder.setup_agent()

    assert thinker.agent_sdk == "openai"
    assert any(isinstance(tool, fake_shell_tool) for tool in thinker.agent.kwargs["tools"])
    assert any(
        isinstance(tool, fake_shell_tool) for tool in writer.write_agent.kwargs["tools"]
    )
    assert any(
        isinstance(tool, fake_shell_tool) for tool in reviewer.review_agent.kwargs["tools"]
    )
    assert any(isinstance(tool, fake_shell_tool) for tool in coder.agent.kwargs["tools"])


def test_claude_sdk_smoke_wires_generated_mcp(monkeypatch, tmp_path: Path) -> None:
    _install_fake_agents()
    _install_fake_runtime_modules()
    _clear_tiny_scientist_modules()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")

    thinker_module = importlib.import_module("tiny_scientist.thinker")
    writer_module = importlib.import_module("tiny_scientist.writer")
    reviewer_module = importlib.import_module("tiny_scientist.reviewer")
    coder_module = importlib.import_module("tiny_scientist.coder")

    thinker = thinker_module.Thinker(
        model="claude-3-5-sonnet-20241022",
        output_dir=str(tmp_path),
        tools=[],
        agent_sdk="claude",
    )
    writer = writer_module.Writer(
        model="claude-3-5-sonnet-20241022",
        output_dir=str(tmp_path),
        template="acl",
        agent_sdk="claude",
    )
    reviewer = reviewer_module.Reviewer(
        model="claude-3-5-sonnet-20241022",
        tools=[],
        agent_sdk="claude",
    )
    coder = coder_module.Coder(
        model="claude-3-5-sonnet-20241022",
        output_dir=str(tmp_path),
        use_docker=False,
        agent_sdk="claude",
    )
    coder.setup_agent()

    assert thinker.agent_sdk == "claude"
    assert thinker.agent.mcp_config_path.endswith(".tiny_scientist.generated.mcp.json")
    assert "mcp__tiny_scientist_research__paper_search" in thinker.agent.allowed_tools
    assert "REFERENCE SKILLS" not in thinker.agent.instructions
    assert writer.write_agent.mcp_config_path.endswith(".tiny_scientist.generated.mcp.json")
    assert "REFERENCE SKILLS" not in writer.write_agent.instructions
    assert reviewer.review_agent.mcp_config_path.endswith(".tiny_scientist.generated.mcp.json")
    assert "REFERENCE SKILLS" not in reviewer.review_agent.instructions
    assert "Bash" in coder.claude_runners["coder"].allowed_tools
    assert "Write" in coder.claude_runners["coder"].allowed_tools
    assert "mcp__tiny_scientist_research__code_search" not in coder.claude_runners["coder"].allowed_tools
    assert "REFERENCE SKILLS" not in coder.claude_runners["coder"].instructions


def test_thinker_builds_default_todo_from_configuration(
    monkeypatch, tmp_path: Path
) -> None:
    _install_fake_agents()
    _install_fake_runtime_modules()
    _clear_tiny_scientist_modules()

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    thinker_module = importlib.import_module("tiny_scientist.thinker")
    thinker = thinker_module.Thinker(
        model="gpt-4o",
        output_dir=str(tmp_path),
        tools=[],
        agent_sdk="openai",
        enable_safety_check=True,
        generate_exp_plan=True,
    )

    todo = thinker._build_todo(check_novelty=False)

    actions = [item["action"] for item in todo]
    assert actions[0] == "generate_idea"
    assert "augment_idea_research" in actions
    assert "experiment_plan" in actions
    assert "safety_check" in actions


def test_coder_blueprint_progress_tracks_current_row(monkeypatch, tmp_path: Path) -> None:
    _install_fake_agents()
    _install_fake_runtime_modules()
    _clear_tiny_scientist_modules()

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    coder_module = importlib.import_module("tiny_scientist.coder")
    coder = coder_module.Coder(
        model="gpt-4o",
        output_dir=str(tmp_path),
        use_docker=False,
        agent_sdk="openai",
    )

    checklist = [
        {
            "step": 1,
            "action": "implement",
            "name": "Load Data",
            "description": "Implement dataset loading",
            "row_refs": ["Dataset"],
        },
        {
            "step": 2,
            "action": "implement",
            "name": "Train Model",
            "description": "Implement training loop",
            "row_refs": ["Training Setup"],
        },
        {
            "step": 3,
            "action": "implement",
            "name": "Evaluate",
            "description": "Compute final metrics",
            "row_refs": ["Evaluation Metrics"],
        },
    ]

    rows = coder._build_blueprint_progress_rows(
        checklist=checklist,
        table_rows=["Dataset", "Training Setup", "Evaluation Metrics"],
        completed_steps=1,
        active_step=checklist[1],
    )

    assert rows[0]["status"] == "completed"
    assert rows[1]["status"] == "in_progress"
    assert rows[2]["status"] == "pending"


def test_reviewer_builds_default_todo_from_configuration(
    monkeypatch, tmp_path: Path
) -> None:
    _install_fake_agents()
    _install_fake_runtime_modules()
    _clear_tiny_scientist_modules()

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    reviewer_module = importlib.import_module("tiny_scientist.reviewer")
    reviewer = reviewer_module.Reviewer(
        model="gpt-4o",
        tools=[],
        agent_sdk="openai",
        num_reflections=1,
    )

    todo = reviewer._build_todo()

    actions = [item["action"] for item in todo]
    assert actions[0] == "generate_review"
    assert "reflect_review" in actions
    assert actions[-1] == "meta_review"


def test_thinker_extracts_citations_from_key_reference_urls(
    monkeypatch, tmp_path: Path
) -> None:
    _install_fake_agents()
    _install_fake_runtime_modules()
    _clear_tiny_scientist_modules()

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    thinker_module = importlib.import_module("tiny_scientist.thinker")
    thinker = thinker_module.Thinker(
        model="gpt-4o",
        output_dir=str(tmp_path),
        tools=[],
        agent_sdk="openai",
    )

    citations = thinker._extract_citations_with_urls(
        {
            "model_candidates": [
                {
                    "name": "Gradient Descent",
                    "key_reference_urls": ["https://doi.org/10.1137/1011036"],
                }
            ]
        }
    )

    assert citations
    assert citations[0]["url"] == "https://doi.org/10.1137/1011036"


def test_thinker_extracts_citations_from_raw_text_when_json_is_broken(
    monkeypatch, tmp_path: Path
) -> None:
    _install_fake_agents()
    _install_fake_runtime_modules()
    _clear_tiny_scientist_modules()

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    thinker_module = importlib.import_module("tiny_scientist.thinker")
    thinker = thinker_module.Thinker(
        model="gpt-4o",
        output_dir=str(tmp_path),
        tools=[],
        agent_sdk="openai",
    )

    citations = thinker._extract_citations_from_text(
        '{\n'
        '  "model_candidates": [\n'
        '    {"name": "Gradient Descent", "key_reference_urls": ["https://doi.org/10.1137/1011036"]}\n'
        "  ]\n"
    )

    assert citations
    assert citations[0]["title"] == "Gradient Descent"
    assert citations[0]["url"] == "https://doi.org/10.1137/1011036"
