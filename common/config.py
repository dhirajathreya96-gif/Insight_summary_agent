
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict

@dataclass
class InsightAgentConfig:
    # What to include
    top_n: int = 5
    priority_categories: List[str] = field(default_factory=list)  # optional, matches "Speciality"
    priority_skus: List[str] = field(default_factory=list)        # matches SKU_Name/Product_Name
    own_brands: List[str] = field(default_factory=list)           # used to separate competitor vs own
    competitor_brands: List[str] = field(default_factory=list)

    # Comparison behavior
    price_change_threshold_pct: float = 1.0   # report if abs % change >= this
    promotion_change_threshold_pct: float = 1.0

    # Filters
    require_serviced: bool = True             # only count rows with Service_Status == "serviced"

    # Delivery
    recipients_email: List[str] = field(default_factory=list)
    recipients_whatsapp: List[str] = field(default_factory=list)  # E.164 format, e.g. +9199...
    subject_prefix: str = "[Spoggle]"

    # Misc
    timezone: str = "America/Chicago"

    def to_dict(self) -> Dict:
        return {
            "top_n": self.top_n,
            "priority_categories": self.priority_categories,
            "priority_skus": self.priority_skus,
            "own_brands": self.own_brands,
            "competitor_brands": self.competitor_brands,
            "price_change_threshold_pct": self.price_change_threshold_pct,
            "promotion_change_threshold_pct": self.promotion_change_threshold_pct,
            "require_serviced": self.require_serviced,
            "recipients_email": self.recipients_email,
            "recipients_whatsapp": self.recipients_whatsapp,
            "subject_prefix": self.subject_prefix,
            "timezone": self.timezone,
        }
