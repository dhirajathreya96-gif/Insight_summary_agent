# common/action_config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ActionAgentConfig:
    """
    Stage 2 config (Action Agent).

    This layer is intentionally isolated from Stage 1 so that:
      - Stage 1 logic/output stays unchanged
      - Stage 2 can evolve rapidly (rules -> LLM -> hybrid)
    """

    top_n: int = 5

    # Severity thresholds for stockout classification
    alert_stockout_oos_pct_high: float = 90.0
    alert_stockout_oos_pct_medium: float = 70.0

    # Severity thresholds for delta vs previous period (pp = percentage points)
    alert_delta_oos_pp_high: float = 5.0
    alert_delta_oos_pp_medium: float = 2.0

    # ✅ Price / promo thresholds (referenced by action_agent.py gating)
    alert_price_change_pct_high: float = 5.0
    alert_disc_change_pp_high: float = 5.0
    alert_disc_pct_high: float = 30.0

    # ✅ Reporting filter: do not show low/zero OOS items in Stage 2 evidence
    min_oos_pct_for_reporting: float = 50.0

    # What you want to focus on in Stage 2
    preferred_alert_types: List[str] = None  # e.g. ["stockouts","price_promo","competition"]
    suppression_rules: List[str] = None      # free-form simple rules (string list)
    role_views: List[str] = None             # e.g. ["Supply Chain","E-Commerce Ops","Sales"]

    # LLM toggles
    use_llm: bool = True
    llm_model: str = "gpt-4o-mini"

    def __post_init__(self):
        if self.preferred_alert_types is None:
            self.preferred_alert_types = ["stockouts", "price_promo", "competition"]
        if self.suppression_rules is None:
            self.suppression_rules = []
        if self.role_views is None:
            self.role_views = []
