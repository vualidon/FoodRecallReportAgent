from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class RecallSource(str, Enum):
    """Source of the food recall data."""
    FDA = "FDA"
    USDA = "USDA"


class HealthRisk(str, Enum):
    """Severity of health risk associated with the recall."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class DistributionScope(str, Enum):
    """Geographic scope of product distribution."""
    LOCAL = "local"
    REGIONAL = "regional"
    NATIONAL = "national"
    INTERNATIONAL = "international"
    UNKNOWN = "unknown"


class EconomicImpact(str, Enum):
    """Estimated economic impact of the recall."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class RawRecallData(BaseModel):
    """Raw data collected from recall sources before processing."""
    source: RecallSource
    url: str
    html_content: str
    collected_at: datetime = Field(default_factory=datetime.now)


class FoodRecall(BaseModel):
    """Structured food recall information after extraction."""
    id: str
    source: RecallSource
    url: str
    title: str
    product_name: str
    brand_name: Optional[str] = None
    recalling_firm: Optional[str] = None
    recall_date: Optional[datetime] = None
    reason: str
    health_risk: HealthRisk = HealthRisk.UNKNOWN
    distribution_scope: DistributionScope = DistributionScope.UNKNOWN
    distribution_states: Optional[List[str]] = None
    lot_codes: Optional[List[str]] = None
    economic_impact: EconomicImpact = EconomicImpact.UNKNOWN
    impact_score: Optional[float] = None
    analyzed_at: Optional[datetime] = None 