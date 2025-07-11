from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from .configs import Config
from .utils.budget_checker import BudgetChecker
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
    """Safety checker module for Tiny Scientist"""

    def __init__(
        self, model: str = "gpt-4o", cost_tracker: Optional[BudgetChecker] = None
    ) -> None:
        self.model = model
        self.cost_tracker = cost_tracker or BudgetChecker()
        self.detector = PromptAttackDetector(
            model=model, cost_tracker=self.cost_tracker
        )

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
