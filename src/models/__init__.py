"""
Data models for the Food Recall Report System.

Contains Pydantic models used for structured data representation across the system.
"""

from src.models.food_recall import (
    FoodRecall,
    RawRecallData,
    RecallSource,
    HealthRisk,
    DistributionScope,
    EconomicImpact
)

__all__ = [
    'FoodRecall',
    'RawRecallData',
    'RecallSource',
    'HealthRisk',
    'DistributionScope',
    'EconomicImpact'
] 