#!/usr/bin/env python3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import os
import time
import json
from datetime import datetime

ROOT = Path("/Users/yuhaofei/Downloads/all_results_full/tinyscientist_nonbio_wotool")
MODEL = "gpt-4o"
WORKERS = int(os.getenv("WORKERS", "5"))  # 并行度，改环境变量即可

# ============================================================================
# PROMPT 1: Writing Quality Assessment
# ============================================================================
WRITING_QUALITY_PROMPT = """You are an expert AI research reviewer evaluating the WRITING QUALITY and PRESENTATION of a research paper.

**Paper Content:**
{paper_text}

**Your Task - Evaluate Writing Quality:**
  Quality Rubric (1–5 scale) focusing on Content Richness, References, and Writing:

  Score 1 — Very Poor: Critical information missing or wrong; methodology clearly infeasible or 
  incoherent.
    Examples: Missing problem statement, no clear methodology, completely unrealistic approach, 
              no literature review, no references to recent work.

  Score 2 — Poor: Key sections under-specified and novelty minimal; noticeable feasibility 
  or consistency issues.
    Examples: Vague problem description, weak novelty claim, unclear methodology, limited scope, 
              few or outdated references, minimal technical depth.

  Score 2.5 — Below Average: Basic structure present but lacks depth and recent references.
    Examples: Standard approach with minimal innovation, basic methodology, some references but not 
    cutting-edge, 
              feasible but not compelling, limited technical sophistication.

  Score 3 — Average: All required fields present and executable, but largely routine with limited 
  depth or insight.
    Examples: Standard approach, basic methodology, some novelty but not compelling, feasible but not 
    innovative, 
              references to well-known but not recent work.

  Score 3.5 — Above Average: Well-structured with some depth and recent references.
    Examples: Clear problem and methodology, some novel insights, references to recent papers 
    (2020-2023), 
              good feasibility, moderate technical sophistication.

  Score 4 — Good: Well-structured, mostly complete, shows some innovation; only minor gaps or risks 
  remain.
    Examples: Clear problem and methodology, some novel insights, well-defined scope, good feasibility, 
              references to recent and relevant work, good technical depth.

  Score 4.5 — Very Good: Strong structure with significant innovation and recent references.
    Examples: Highly innovative approach, comprehensive methodology, strong novelty, clear impact 
    potential, 
              references to cutting-edge work (2022-2024), excellent technical depth.

  Score 5 — Excellent: Original, rigorous, and comprehensive; strong experimental design and clear 
  path to publication.
    Examples: Highly innovative approach, comprehensive methodology, strong novelty, clear 
    impact potential, 
              references to state-of-the-art work, exceptional technical sophistication.

  Key Evaluation Criteria for Writing Quality:
    1. Content Richness: Depth of problem analysis, methodology detail, technical sophistication
    2. Reference Quality: Use of recent, relevant, and cutting-edge references (2020-2024 preferred)
    3. Writing Clarity: Clear exposition, logical flow, readability
    4. Technical Depth: Thoroughness of technical description and analysis
    5. Completeness: All necessary sections present and well-developed

**Output Format:**
Please respond with ONLY a JSON object:
{{
  "score": <number between 1 and 5, can use 0.5 increments>,
  "reason": "<detailed explanation focusing on writing quality, content depth, and references>"
}}
"""

# ============================================================================
# PROMPT 2: Research Value Assessment
# ============================================================================
RESEARCH_VALUE_PROMPT = """You are an expert AI research reviewer evaluating the RESEARCH VALUE and CONTRIBUTION of a research paper.

**Paper Content:**
{paper_text}

**Your Task - Evaluate Research Value:**
  Value Rubric (1–5 scale) focusing on Feasibility, Novelty, and Usefulness:

  Score 1 — Very Poor: Not feasible, no novelty, not useful.
    Examples: Unrealistic approach, duplicates existing work, provides no value to researchers.

  Score 2 — Poor: Major feasibility issues, minimal novelty, limited usefulness.
    Examples: Difficult to implement, minor variations of existing work, marginal contribution.

  Score 2.5 — Below Average: Some feasibility concerns, incremental novelty, modest usefulness.
    Examples: Technically possible but challenging, small improvements, some utility for specific cases.

  Score 3 — Average: Feasible with effort, moderate novelty, decent usefulness.
    Examples: Standard approach with reasonable implementation, combines existing ideas in new ways, 
              useful for a subset of researchers.

  Score 3.5 — Above Average: Clearly feasible, good novelty, quite useful.
    Examples: Practical to implement, introduces interesting new perspectives, valuable for many 
              researchers in the field.

  Score 4 — Good: Highly feasible, significant novelty, very useful.
    Examples: Easy to implement and adopt, introduces novel techniques or insights, addresses 
              important problems, valuable for most researchers.

  Score 4.5 — Very Good: Excellent feasibility, strong novelty, highly useful.
    Examples: Ready-to-use solution, groundbreaking insights, solves critical problems, 
              game-changing for the field.

  Score 5 — Excellent: Perfect feasibility, exceptional novelty, transformative usefulness.
    Examples: Plug-and-play solution, paradigm-shifting ideas, revolutionizes the field, 
              essential for all researchers.

  Key Evaluation Criteria for Research Value:
    1. Feasibility: How practical and implementable is this research?
    2. Novelty: How original and innovative is the contribution?
    3. Usefulness: How valuable is this for researchers? Does it solve real problems?
    4. Impact Potential: Will this influence future research?
    5. Reproducibility: Can others easily build upon this work?

**Output Format:**
Please respond with ONLY a JSON object:
{{
  "score": <number between 1 and 5, can use 0.5 increments>,
  "reason": "<detailed explanation focusing on feasibility, novelty, and usefulness for researchers>"
}}
"""

WRITING_SYSTEM_PROMPT = "You are an expert peer reviewer evaluating writing quality, content richness, and presentation. Focus on how well the paper is written, organized, and documented."

RESEARCH_SYSTEM_PROMPT = "You are an expert peer reviewer evaluating research value and contribution. Focus on feasibility, novelty, and practical usefulness for researchers."

def _to_jsonable(res):
    """Convert LLM response to JSON, expecting {score, reason} format."""
    try:
        if hasattr(res, "to_dict") and callable(getattr(res, "to_dict")):
            return res.to_dict()
        if isinstance(res, (dict, list)):
            return res
        if isinstance(res, str):
            # Try to parse as JSON first
            try:
                res = res.strip().strip("`").strip("json").strip()
                parsed = json.loads(res)
                # Validate expected format
                if isinstance(parsed, dict) and "score" in parsed and "reason" in parsed:
                    return parsed
                else:
                    # If not in expected format, wrap it
                    return {"score": None, "reason": str(parsed)}
            except json.JSONDecodeError:
                # If not valid JSON, try to extract JSON from markdown code blocks
                import re
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', res, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(1))
                        if isinstance(parsed, dict) and "score" in parsed and "reason" in parsed:
                            return parsed
                    except:
                        pass
                # Fallback: return raw text
                return {"score": None, "reason": res}
        return {"score": None, "reason": str(res)}
    except Exception as e:
        return {"score": None, "reason": str(res), "error": str(e)}

def review_one(report_path: Path) -> tuple[str, float | None, float | None]:
    """
    Review one paper with TWO scores and return (status_message, writing_score, research_score).
    Returns (message, writing_score, research_score) where scores are None if failed.
    """
    from tiny_scientist.utils.llm import create_client, get_response_from_llm
    
    # 创建 LLM 客户端
    client, model = create_client(MODEL)
    
    # 读取论文内容
    raw_text = report_path.read_text(encoding="utf-8", errors="ignore")

    writing_score = None
    research_score = None
    
    for attempt in range(3):
        try:
            # ===== EVALUATION 1: Writing Quality =====
            writing_prompt = WRITING_QUALITY_PROMPT.format(paper_text=raw_text)
            writing_result, _ = get_response_from_llm(
                msg=writing_prompt,
                client=client,
                model=model,
                system_message=WRITING_SYSTEM_PROMPT,
                cost_tracker=None,
                task_name="writing_quality_eval"
            )
            writing_data = _to_jsonable(writing_result)
            writing_score = writing_data.get("score")
            
            # ===== EVALUATION 2: Research Value =====
            research_prompt = RESEARCH_VALUE_PROMPT.format(paper_text=raw_text)
            research_result, _ = get_response_from_llm(
                msg=research_prompt,
                client=client,
                model=model,
                system_message=RESEARCH_SYSTEM_PROMPT,
                cost_tracker=None,
                task_name="research_value_eval"
            )
            research_data = _to_jsonable(research_result)
            research_score = research_data.get("score")
            
            # 保存结果
            payload = {
                "model": MODEL,
                "source_txt": str(report_path),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "writing_quality": {
                    "score": writing_score,
                    "reason": writing_data.get("reason"),
                    "raw_result": writing_result
                },
                "research_value": {
                    "score": research_score,
                    "reason": research_data.get("reason"),
                    "raw_result": research_result
                }
            }
            out_path = report_path.parent / "quality_eval.json"
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            
            # 打印评分信息
            writing_preview = writing_data.get("reason", "")[:80] + "..." if writing_data.get("reason") and len(writing_data.get("reason", "")) > 80 else writing_data.get("reason", "N/A")
            research_preview = research_data.get("reason", "")[:80] + "..." if research_data.get("reason") and len(research_data.get("reason", "")) > 80 else research_data.get("reason", "N/A")
            msg = f"[OK] {out_path}\n  📝 Writing: {writing_score} | {writing_preview}\n  🔬 Research: {research_score} | {research_preview}"
            return (msg, writing_score, research_score)
            
        except Exception as e:
            if attempt == 2:
                return (f"[ERROR] {report_path}: {e}", None, None)
            time.sleep(1.5 * (attempt + 1))
    
    return (f"[ERROR] {report_path}: Failed after retries", None, None)

def main():
    # 查找所有可能的目录
    all_dirs = sorted([d for d in ROOT.iterdir() if d.is_dir()])
    
    # 查找存在的报告文件
    report_files = sorted(ROOT.glob("*/latex/acl_latex.tex"))
    
    # 找出缺失报告的目录
    dirs_with_reports = {f.parent.parent for f in report_files}
    missing_report_dirs = [d for d in all_dirs if d not in dirs_with_reports]
    
    if not report_files:
        print(f"[WARN] no acl_latex.tex under immediate subfolders of {ROOT}")
        return

    print(f"[INFO] Total directories: {len(all_dirs)}")
    print(f"[INFO] Found {len(report_files)} reports")
    if missing_report_dirs:
        print(f"[WARN] Missing reports in {len(missing_report_dirs)} directories:")
        for missing_dir in missing_report_dirs[:10]:  # 只显示前10个
            print(f"  - {missing_dir.name}")
        if len(missing_report_dirs) > 10:
            print(f"  ... and {len(missing_report_dirs) - 10} more")
    print(f"[INFO] Workers={WORKERS}")
    print("=" * 80)
    
    writing_scores = []
    research_scores = []
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        fut2file = {ex.submit(review_one, p): p for p in report_files}
        for fut in as_completed(fut2file):
            msg, writing_score, research_score = fut.result()
            print(msg)
            
            # Collect scores
            writing_valid = False
            research_valid = False
            
            if writing_score is not None:
                try:
                    writing_scores.append(float(writing_score))
                    writing_valid = True
                except (ValueError, TypeError):
                    pass
            
            if research_score is not None:
                try:
                    research_scores.append(float(research_score))
                    research_valid = True
                except (ValueError, TypeError):
                    pass
            
            if not (writing_valid and research_valid):
                failed_count += 1
    
    # 打印统计信息
    print("=" * 80)
    print("\n📊 EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total directories: {len(all_dirs)}")
    print(f"Total papers found: {len(report_files)}")
    print(f"Missing reports: {len(missing_report_dirs)}")
    print(f"Successfully evaluated: {min(len(writing_scores), len(research_scores))}")
    print(f"Failed evaluations: {failed_count}")
    
    if writing_scores and research_scores:
        # Writing Quality Statistics
        avg_writing = sum(writing_scores) / len(writing_scores)
        min_writing = min(writing_scores)
        max_writing = max(writing_scores)
        median_writing = sorted(writing_scores)[len(writing_scores)//2]
        
        # Research Value Statistics
        avg_research = sum(research_scores) / len(research_scores)
        min_research = min(research_scores)
        max_research = max(research_scores)
        median_research = sorted(research_scores)[len(research_scores)//2]
        
        print(f"\n📝 WRITING QUALITY STATISTICS")
        print(f"Average: {avg_writing:.2f} | Min: {min_writing:.1f} | Max: {max_writing:.1f} | Median: {median_writing:.1f}")
        
        print(f"\n🔬 RESEARCH VALUE STATISTICS")
        print(f"Average: {avg_research:.2f} | Min: {min_research:.1f} | Max: {max_research:.1f} | Median: {median_research:.1f}")
        
        # Score distributions
        print(f"\n📊 WRITING QUALITY DISTRIBUTION")
        writing_counts = Counter(writing_scores)
        for score in sorted(writing_counts.keys()):
            count = writing_counts[score]
            bar = "█" * count
            print(f"  {score:.1f}: {bar} ({count})")
        
        print(f"\n📊 RESEARCH VALUE DISTRIBUTION")
        research_counts = Counter(research_scores)
        for score in sorted(research_counts.keys()):
            count = research_counts[score]
            bar = "█" * count
            print(f"  {score:.1f}: {bar} ({count})")
        
        # Save summary to file
        summary_path = ROOT / "evaluation_summary.json"
        summary = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": MODEL,
            "total_directories": len(all_dirs),
            "total_papers": len(report_files),
            "missing_reports": len(missing_report_dirs),
            "missing_report_dirs": [d.name for d in missing_report_dirs],
            "successful": min(len(writing_scores), len(research_scores)),
            "failed": failed_count,
            "writing_quality": {
                "average": round(avg_writing, 2),
                "min": min_writing,
                "max": max_writing,
                "median": median_writing,
                "all_scores": writing_scores,
                "distribution": dict(writing_counts)
            },
            "research_value": {
                "average": round(avg_research, 2),
                "min": min_research,
                "max": max_research,
                "median": median_research,
                "all_scores": research_scores,
                "distribution": dict(research_counts)
            }
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n💾 Summary saved to: {summary_path}")
        
        # Save detailed missing reports list
        if missing_report_dirs:
            missing_path = ROOT / "missing_reports.txt"
            missing_path.write_text(
                "\n".join([d.name for d in missing_report_dirs]),
                encoding="utf-8"
            )
            print(f"📝 Missing reports list saved to: {missing_path}")
    else:
        print("\n⚠️  No valid scores collected!")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
