#!/usr/bin/env python3
import argparse
import json
import os
from glob import glob
from typing import Dict, List, Optional, Any

import difflib
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze attack logs from malicious agents")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./output/attack_logs",
        help="Directory containing attack log files",
    )
    parser.add_argument(
        "--session_id",
        type=str,
        default=None,
        help="Specific session ID to analyze (if None, analyze all)",
    )
    parser.add_argument(
        "--detail_level",
        type=str,
        choices=["summary", "medium", "full"],
        default="medium",
        help="Level of detail in analysis output",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save analysis to file instead of printing to console",
    )
    
    return parser.parse_args()


def load_attack_logs(log_dir: str, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load attack logs from the specified directory."""
    logs = []
    
    # Get list of log files
    if session_id:
        log_files = glob(os.path.join(log_dir, f"{session_id}.json"))
    else:
        log_files = glob(os.path.join(log_dir, "*.json"))
    
    if not log_files:
        print(f"No log files found in {log_dir}" + (f" for session {session_id}" if session_id else ""))
        return []
    
    # Load each log file
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
                logs.append(log_data)
        except Exception as e:
            print(f"Error loading log file {log_file}: {e}")
    
    return logs


def get_diff_highlights(original: str, manipulated: str) -> List[str]:
    """Generate color-highlighted diff between original and manipulated texts."""
    differ = difflib.Differ()
    diff = list(differ.compare(original.splitlines(), manipulated.splitlines()))
    
    highlights = []
    for line in diff:
        if line.startswith("  "):  # unchanged
            highlights.append(line[2:])
        elif line.startswith("- "):  # removed
            highlights.append(f"[red]{line[2:]}[/red]")
        elif line.startswith("+ "):  # added
            highlights.append(f"[green]{line[2:]}[/green]")
    
    return highlights


def analyze_log(log: Dict[str, Any], detail_level: str) -> List[Dict[str, Any]]:
    """Analyze a single attack log."""
    analysis_results = []
    
    # Extract basic information
    session_id = log.get("session_id", "unknown")
    intent = log.get("intent", "")
    domain = log.get("domain", "")
    attack_severity = log.get("attack_severity", "unknown")
    
    # Analyze each intercepted message
    intercepted_messages = log.get("intercepted_messages", {})
    for msg_id, msg_data in intercepted_messages.items():
        if "manipulated" not in msg_data:
            continue  # Skip messages that weren't manipulated
            
        original = msg_data.get("original", {})
        manipulated = msg_data.get("manipulated", {})
        attack_agent = msg_data.get("attack_agent", "unknown")
        attack_target = msg_data.get("attack_target", "unknown")
        
        # Generate diffs
        thought_diff = get_diff_highlights(original.get("thought", ""), manipulated.get("thought", ""))
        suggestions_diff = get_diff_highlights(original.get("suggestions", ""), manipulated.get("suggestions", ""))
        rating_diff = f"Original: {original.get('rating', 'N/A')} → Manipulated: {manipulated.get('rating', 'N/A')}"
        
        analysis_results.append({
            "session_id": session_id,
            "message_id": msg_id,
            "agent": original.get("agent", "unknown"),
            "role": original.get("role", "unknown"),
            "attack_agent": attack_agent,
            "attack_target": attack_target,
            "thought_diff": thought_diff,
            "suggestions_diff": suggestions_diff,
            "rating_diff": rating_diff,
            "original": original,
            "manipulated": manipulated
        })
    
    return analysis_results


def display_analysis_results(results: List[Dict[str, Any]], detail_level: str, console: Console) -> None:
    """Display analysis results with the specified level of detail."""
    if not results:
        console.print("[yellow]No manipulation detected in the logs.[/yellow]")
        return
    
    # Group results by session
    results_by_session = {}
    for result in results:
        session_id = result["session_id"]
        if session_id not in results_by_session:
            results_by_session[session_id] = []
        results_by_session[session_id].append(result)
    
    # Display results for each session
    for session_id, session_results in results_by_session.items():
        console.print(f"\n[bold]===== Session: {session_id} =====")
        console.print(f"Total manipulations: {len(session_results)}")
        
        # Create a summary table
        table = Table(title=f"Manipulation Summary for Session {session_id}")
        table.add_column("Message ID", style="cyan")
        table.add_column("Target Agent", style="magenta")
        table.add_column("Attacker", style="red")
        table.add_column("Target Aspect", style="yellow")
        table.add_column("Rating Change", style="green")
        
        for result in session_results:
            table.add_row(
                result["message_id"],
                f"{result['role']} ({result['agent']})",
                result["attack_agent"],
                result["attack_target"],
                result["rating_diff"]
            )
        
        console.print(table)
        
        # Display detailed diffs if requested
        if detail_level in ["medium", "full"]:
            for i, result in enumerate(session_results):
                console.print(f"\n[bold cyan]Manipulation {i+1}:[/bold cyan]")
                console.print(f"Message: {result['message_id']}")
                console.print(f"Target: {result['role']} ({result['agent']})")
                console.print(f"Attacker: {result['attack_agent']}")
                console.print(f"Target Aspect: {result['attack_target']}")
                
                if detail_level == "full":
                    console.print("\n[bold]Original Thought:[/bold]")
                    console.print(result["original"]["thought"])
                    console.print("\n[bold]Manipulated Thought:[/bold]")
                    console.print(result["manipulated"]["thought"])
                    
                    console.print("\n[bold]Original Suggestions:[/bold]")
                    console.print(result["original"]["suggestions"])
                    console.print("\n[bold]Manipulated Suggestions:[/bold]")
                    console.print(result["manipulated"]["suggestions"])
                else:
                    # Medium detail - show diff highlights
                    console.print("\n[bold]Thought Changes:[/bold]")
                    for line in result["thought_diff"]:
                        console.print(line, highlight=False)
                    
                    console.print("\n[bold]Suggestions Changes:[/bold]")
                    for line in result["suggestions_diff"]:
                        console.print(line, highlight=False)
                
                console.print("\n[bold]Rating Change:[/bold]")
                console.print(result["rating_diff"])
                
                if i < len(session_results) - 1:
                    console.print("\n" + "-" * 50)


def main():
    """Run the attack log analysis."""
    args = parse_args()
    
    # Load attack logs
    logs = load_attack_logs(args.log_dir, args.session_id)
    if not logs:
        print("No logs to analyze.")
        return
    
    # Create console for output
    console = Console(file=open(args.output, "w") if args.output else None)
    
    # Analyze each log
    all_results = []
    for log in logs:
        results = analyze_log(log, args.detail_level)
        all_results.extend(results)
    
    # Display results
    console.print(f"[bold]Analysis of {len(logs)} Attack Sessions[/bold]")
    display_analysis_results(all_results, args.detail_level, console)
    
    if args.output:
        print(f"Analysis saved to {args.output}")
        console.file.close()


if __name__ == "__main__":
    main() 