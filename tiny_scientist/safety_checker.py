import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import yaml

from .budget_checker import BudgetChecker
from .configs import Config
from .utils.error_handler import api_calling_error_exponential_backoff
from .utils.llm import create_client, get_response_from_llm


class RiskLevel(Enum):
    BLOCK = "block"
    WARNING = "warning"
    SAFE = "safe"
    NULL = None


@dataclass
class DetectionResult:
    risk_level: RiskLevel
    risk_reason: str
    attack_type: Optional[str] = None


def is_rejection_response(response: str) -> bool:
    """Check if the response is a direct rejection"""
    rejection_phrases = [
        "i'm sorry",
        "i can't",
        "i cannot",
        "i am unable",
        "i am not able",
        "i don't",
        "i do not",
        "i won't",
        "i will not",
    ]
    response_lower = response.lower()
    return any(phrase in response_lower for phrase in rejection_phrases)


class PromptAttackDetector:
    def __init__(
        self, model: str = "gpt-4o", cost_tracker: Optional[BudgetChecker] = None
    ) -> None:
        self.client, self.model = create_client(model)
        self.cost_tracker = cost_tracker or BudgetChecker()
        self.config = Config()
        self.prompts = self.config.prompt_template.safety_prompt

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def assess_risk_with_llm(self, prompt: str) -> Tuple[RiskLevel, Optional[str]]:
        """Use LLM to assess the risk level of the prompt"""
        try:
            text, _ = get_response_from_llm(
                prompt,
                client=self.client,
                model=self.model,
                system_message=self.prompts.risk_assessment_system_prompt,
                msg_history=[],
                temperature=0.3,
                cost_tracker=self.cost_tracker,
                task_name="assess_risk",
            )

            if text is None or is_rejection_response(text):
                return RiskLevel.NULL, None

            try:
                risk_level_str = text.split("RISK_LEVEL:")[1].split("\n")[0].strip()
                reason = text.split("REASON:")[1].strip()
                return RiskLevel[risk_level_str], reason
            except (IndexError, KeyError) as e:
                print(f"Error parsing risk assessment response: {str(e)}")
                print(f"Raw response: {text}")
                return RiskLevel.NULL, None

        except Exception as e:
            print(f"Error in LLM assessment: {str(e)}")
            return RiskLevel.NULL, None

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def detect_attack_with_llm(
        self, prompt: str
    ) -> Tuple[Optional[bool], Optional[str]]:
        """Use LLM to detect potential prompt attacks"""
        try:
            text, _ = get_response_from_llm(
                prompt,
                client=self.client,
                model=self.model,
                system_message=self.prompts.attack_detection_system_prompt,
                msg_history=[],
                temperature=0.3,
                cost_tracker=self.cost_tracker,
                task_name="detect_attack",
            )

            if text is None or is_rejection_response(text):
                return None, None

            try:
                is_attacked = (
                    text.split("IS_ATTACKED:")[1].split("\n")[0].strip().lower()
                    == "true"
                )
                attack_type = text.split("ATTACK_TYPE:")[1].split("\n")[0].strip()
                return is_attacked, attack_type if is_attacked else "None"
            except (IndexError, KeyError) as e:
                print(f"Error parsing attack detection response: {str(e)}")
                print(f"Raw response: {text}")
                return None, None

        except Exception as e:
            print(f"Error in attack detection: {str(e)}")
            return None, None

    def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        risk_level, risk_reason = self.assess_risk_with_llm(prompt)

        is_attacked, attack_type = self.detect_attack_with_llm(prompt)

        output = {
            "risk_assessment": {
                "level": risk_level.value if risk_level != RiskLevel.NULL else None,
                "reason": risk_reason,
            },
            "attack_detection": {
                "is_attacked": is_attacked,
                "attack_type": attack_type,
            },
        }

        return output


class SafetyChecker:
    """
    Comprehensive safety checker module for Tiny Scientist.

    This class provides:
    1. Intent safety checking (prompt attacks, risk assessment)
    2. Idea ethics evaluation and enhancement
    3. Comprehensive safety analysis combining all checks
    """

    def __init__(
        self, model: str = "gpt-4o", cost_tracker: Optional[BudgetChecker] = None
    ) -> None:
        self.model = model
        self.cost_tracker = cost_tracker or BudgetChecker()
        self.detector = PromptAttackDetector(
            model=model, cost_tracker=self.cost_tracker
        )
        self.client, _ = create_client(self.model)
        self.config = Config()
        # Load ethics prompts
        self.ethics_prompts = self._load_ethics_prompts()

    def check_safety(self, intent: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check the safety of the input intent

        Args:
            intent: The research intent to be checked

        Returns:
            Tuple of (is_safe: bool, safety_report: Dict)
        """
        print("🔒 Performing safety check...")

        # Analyze the prompt for safety
        safety_report = self.detector.analyze_prompt(intent)

        # Determine if the intent is safe to proceed
        risk_level = safety_report["risk_assessment"]["level"]
        is_attacked = safety_report["attack_detection"]["is_attacked"]

        is_safe = True
        if risk_level == "block":
            is_safe = False
            print("❌ BLOCKED: Input poses significant risks")
        elif risk_level == "warning":
            print("⚠️  WARNING: Input has some risks but may proceed with caution")
        elif is_attacked:
            is_safe = False
            print("❌ BLOCKED: Potential prompt attack detected")
        else:
            print("✅ SAFE: Input passed safety checks")

        print(f"Risk Level: {risk_level}")
        print(f"Attack Detected: {is_attacked}")
        if safety_report["risk_assessment"]["reason"]:
            print(f"Reason: {safety_report['risk_assessment']['reason']}")

        self.cost_tracker.report()
        return is_safe, safety_report

    def _load_ethics_prompts(self) -> Dict[str, Any]:
        """Load ethics prompts from the YAML file."""
        prompt_path = os.path.join(
            os.path.dirname(__file__), "prompts", "safety_prompt.yaml"
        )
        try:
            with open(prompt_path, "r", encoding="utf-8") as file:
                import typing

                return typing.cast(Dict[str, Any], yaml.safe_load(file))
        except FileNotFoundError:
            print(f"Warning: Ethics prompts file not found at {prompt_path}")
            return {}

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=2)
    def evaluate_ethics(self, intent: str, idea: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the ethical aspects of a research idea.

        Args:
            intent: The original research intent
            idea: The research idea to evaluate

        Returns:
            Dictionary containing ethical analysis and enhanced idea
        """
        print("🔍 Performing ethics evaluation...")

        if not self.ethics_prompts:
            return {
                "ethical_analysis": "Ethics prompts not available",
                "enhanced_idea": idea,
                "status": "warning",
                "error_message": "Ethics evaluation unavailable",
            }

        try:
            # Format the prompt with the provided data
            formatted_prompt = self.ethics_prompts["ethical_defense_prompt"].format(
                intent=intent, idea=str(idea)
            )

            # Get response from LLM
            response, _ = get_response_from_llm(
                formatted_prompt,
                client=self.client,
                model=self.model,
                system_message=self.ethics_prompts["ethical_defense_system_prompt"],
                msg_history=[],
                temperature=0.3,
                cost_tracker=self.cost_tracker,
                task_name="ethics_evaluation",
            )

            if response is None or is_rejection_response(response):
                return {
                    "ethical_analysis": "Ethics evaluation was rejected",
                    "enhanced_idea": idea,
                    "status": "warning",
                }

            # Parse the response
            parsed_response = self._parse_ethics_response(response)

            print("✅ Ethics evaluation completed")
            self.cost_tracker.report()
            return parsed_response

        except Exception as e:
            print(f"❌ Error in ethics evaluation: {str(e)}")
            return {
                "ethical_analysis": f"Error in ethics evaluation: {str(e)}",
                "enhanced_idea": idea,
                "status": "error",
                "error_message": str(e),
            }

    def _parse_ethics_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the ethics evaluation response.

        Args:
            response: Raw response from LLM

        Returns:
            Parsed response containing analysis and enhanced idea
        """
        try:
            # Split response into analysis and JSON parts
            parts = response.split("ENHANCED IDEA JSON:")

            if len(parts) < 2:
                return {
                    "ethical_analysis": response,
                    "enhanced_idea": None,
                    "status": "warning",
                    "error_message": "Invalid response format",
                }

            analysis_part = parts[0].replace("ETHICAL ANALYSIS:", "").strip()
            json_part = parts[1].strip()

            # Extract JSON from markdown code block
            if "```json" in json_part:
                json_start = json_part.find("```json") + 7
                json_end = json_part.find("```", json_start)
                json_content = json_part[json_start:json_end].strip()
            else:
                json_content = json_part

            # Parse the JSON
            import json

            enhanced_idea = json.loads(json_content)

            return {
                "ethical_analysis": analysis_part,
                "enhanced_idea": enhanced_idea,
                "status": "success",
            }

        except Exception as e:
            return {
                "ethical_analysis": "Error parsing ethics response",
                "enhanced_idea": None,
                "status": "error",
                "error_message": str(e),
            }

    def is_idea_safe(self, idea: Dict[str, Any]) -> bool:
        """
        Quick safety check for a research idea.

        Args:
            idea: The research idea to check

        Returns:
            True if the idea is considered safe, False otherwise
        """
        # This is a simplified safety check
        # In a real implementation, this would involve more sophisticated analysis

        dangerous_keywords = [
            "weapon",
            "harm",
            "attack",
            "exploit",
            "malicious",
            "surveillance",
            "manipulation",
            "deception",
            "toxic",
            "bias",
            "discrimination",
            "privacy violation",
        ]

        idea_text = str(idea).lower()

        for keyword in dangerous_keywords:
            if keyword in idea_text:
                return False

        return True

    def comprehensive_safety_check(
        self, intent: str, idea: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform a comprehensive safety check including both intent safety and idea ethics.

        Args:
            intent: The research intent to check
            idea: Optional research idea to evaluate for ethics

        Returns:
            Dictionary containing comprehensive safety analysis
        """
        print("🔒 Performing comprehensive safety check...")

        # Check intent safety
        is_intent_safe, intent_safety_report = self.check_safety(intent)

        result = {
            "intent_safety": {"is_safe": is_intent_safe, "report": intent_safety_report}
        }

        # Check idea ethics if provided
        if idea is not None:
            ethics_result = self.evaluate_ethics(intent, idea)
            quick_safety = self.is_idea_safe(idea)

            result["idea_ethics"] = {
                "ethics_evaluation": ethics_result,
                "quick_safety_check": quick_safety,
                "is_ethically_sound": ethics_result["status"] == "success"
                and quick_safety,
            }

        # Overall safety determination
        overall_safe = is_intent_safe
        if idea is not None:
            overall_safe = overall_safe and bool(
                result["idea_ethics"]["is_ethically_sound"]
            )

        result["overall_safety"] = {
            "is_safe": overall_safe,
            "recommendation": "PROCEED" if overall_safe else "REVIEW_REQUIRED",
        }

        print(
            f"🎯 Overall safety assessment: {'SAFE' if overall_safe else 'NEEDS REVIEW'}"
        )
        return result
