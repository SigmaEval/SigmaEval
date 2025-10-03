"""
Statistical evaluators for SigmaEval framework.
"""
import logging
from pydantic import BaseModel, Field, field_validator
from typing import List

logger = logging.getLogger("sigmaeval")


class SuccessRateEvaluator(BaseModel):
    """
    Performs one-sided hypothesis test for proportion of successful outcomes.
    
    A rating of 6+ is considered "success", 5 or lower is "failure".
    Tests if true proportion of successes exceeds min_proportion.
    
    Attributes:
        significance_level: Significance level for hypothesis test (e.g., 0.05)
        min_proportion: Minimum acceptable proportion of successes (e.g., 0.90)
        sample_size: Number of samples to collect for statistical analysis
    """
    significance_level: float = Field(
        ..., 
        description="The probability of incorrectly rejecting the null hypothesis (a 'false positive'). Common values are 0.05 (5%) or 0.01 (1%). A value of 0.05 means you accept a 5% chance of concluding the system meets the minimum proportion when it actually doesn't."
    )
    min_proportion: float = Field(..., description="Minimum proportion of successes")
    sample_size: int = Field(..., description="Number of samples to collect")

    @field_validator("significance_level")
    def validate_significance_level(cls, v):
        if not (0 < v < 1):
            raise ValueError("significance_level must be between 0 and 1")
        return v

    @field_validator("min_proportion")
    def validate_min_proportion(cls, v):
        if not (0 <= v <= 1):
            raise ValueError("min_proportion must be between 0 and 1")
        return v

    @field_validator("sample_size")
    def validate_sample_size(cls, v):
        if v <= 0:
            raise ValueError("sample_size must be a positive integer")
        return v

    def evaluate(self, scores: List[float]) -> dict:
        """
        Evaluate if the proportion of successes meets the minimum threshold.
        
        Args:
            scores: List of scores (1-10) from Judge LLM
            
        Returns:
            Dictionary with evaluation results including pass/fail and statistics
        """
        logger.debug(f"SuccessRateEvaluator received scores: {scores}")
        # TODO: Implement one-sided proportion hypothesis test
        # 1. Convert scores to binary (6+ = success, 5 or lower = failure)
        # 2. Calculate observed proportion
        # 3. Perform one-sided hypothesis test
        # 4. Return pass/fail with statistical details
        results = {}
        logger.info(f"SuccessRateEvaluator results: {results}")
        return results


class RatingMeanEvaluator(BaseModel):
    """
    Performs one-sided t-test to determine if mean rating exceeds baseline.
    
    Useful for subjective qualities like helpfulness or tone.
    
    Attributes:
        significance_level: Significance level for hypothesis test (e.g., 0.05)
        min_mean_rating: Minimum acceptable mean rating (e.g., 7.0)
        sample_size: Number of samples to collect for statistical analysis
    """
    significance_level: float = Field(
        ...,
        description="The probability of incorrectly rejecting the null hypothesis (a 'false positive'). Common values are 0.05 (5%) or 0.01 (1%). A value of 0.05 means you accept a 5% chance of concluding the system's mean rating exceeds the minimum when it actually doesn't."
    )
    min_mean_rating: float = Field(..., description="Minimum mean rating threshold")
    sample_size: int = Field(..., description="Number of samples to collect")

    @field_validator("significance_level")
    def validate_significance_level(cls, v):
        if not (0 < v < 1):
            raise ValueError("significance_level must be between 0 and 1")
        return v

    @field_validator("min_mean_rating")
    def validate_min_mean_rating(cls, v):
        if not (1 <= v <= 10):
            raise ValueError("min_mean_rating must be between 1 and 10")
        return v

    @field_validator("sample_size")
    def validate_sample_size(cls, v):
        if v <= 0:
            raise ValueError("sample_size must be a positive integer")
        return v

    def evaluate(self, scores: List[float]) -> dict:
        """
        Evaluate if the mean rating meets the minimum threshold.
        
        Args:
            scores: List of scores (1-10) from Judge LLM
            
        Returns:
            Dictionary with evaluation results including pass/fail and statistics
        """
        logger.debug(f"RatingMeanEvaluator received scores: {scores}")
        # TODO: Implement one-sided t-test
        # 1. Calculate sample mean and standard deviation
        # 2. Perform one-sided t-test against min_mean_rating
        # 3. Return pass/fail with statistical details
        results = {}
        logger.info(f"RatingMeanEvaluator results: {results}")
        return results


class RatingProportionEvaluator(BaseModel):
    """
    Tests if proportion of ratings at or above threshold exceeds minimum.
    
    Similar to SuccessRateEvaluator but with configurable rating threshold.
    
    Attributes:
        significance_level: Significance level for hypothesis test (e.g., 0.05)
        min_rating: Minimum acceptable rating on 1-10 scale (e.g., 8)
        min_proportion: Minimum proportion of responses meeting min_rating (e.g., 0.75)
        sample_size: Number of samples to collect for statistical analysis
    """
    significance_level: float = Field(
        ...,
        description="The probability of incorrectly rejecting the null hypothesis (a 'false positive'). Common values are 0.05 (5%) or 0.01 (1%). A value of 0.05 means you accept a 5% chance of concluding the system meets the minimum proportion when it actually doesn't."
    )
    min_rating: int = Field(..., description="Minimum acceptable rating (1-10)")
    min_proportion: float = Field(..., description="Minimum proportion at or above min_rating")
    sample_size: int = Field(..., description="Number of samples to collect")

    @field_validator("significance_level")
    def validate_significance_level(cls, v):
        if not (0 < v < 1):
            raise ValueError("significance_level must be between 0 and 1")
        return v

    @field_validator("min_rating")
    def validate_min_rating(cls, v):
        if not (1 <= v <= 10):
            raise ValueError("min_rating must be between 1 and 10")
        return v

    @field_validator("min_proportion")
    def validate_min_proportion(cls, v):
        if not (0 <= v <= 1):
            raise ValueError("min_proportion must be between 0 and 1")
        return v

    @field_validator("sample_size")
    def validate_sample_size(cls, v):
        if v <= 0:
            raise ValueError("sample_size must be a positive integer")
        return v

    def evaluate(self, scores: List[float]) -> dict:
        """
        Evaluate if the proportion of ratings meeting threshold is sufficient.
        
        Args:
            scores: List of scores (1-10) from Judge LLM
            
        Returns:
            Dictionary with evaluation results including pass/fail and statistics
        """
        logger.debug(f"RatingProportionEvaluator received scores: {scores}")
        # TODO: Implement one-sided proportion hypothesis test
        # 1. Convert scores to binary (>= min_rating = success, < min_rating = failure)
        # 2. Calculate observed proportion
        # 3. Perform one-sided hypothesis test
        # 4. Return pass/fail with statistical details
        results = {}
        logger.info(f"RatingProportionEvaluator results: {results}")
        return results

