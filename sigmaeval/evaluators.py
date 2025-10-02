"""
Statistical evaluators for SigmaEval framework.
"""

from pydantic import BaseModel, Field
from typing import List


class SuccessRateEvaluator(BaseModel):
    """
    Performs one-sided hypothesis test for proportion of successful outcomes.
    
    A rating of 6+ is considered "success", 5 or lower is "failure".
    Tests if true proportion of successes exceeds min_proportion.
    
    Attributes:
        significance_level: Significance level for hypothesis test (e.g., 0.05)
        min_proportion: Minimum acceptable proportion of successes (e.g., 0.90)
        num_of_samples: Number of samples to collect for statistical analysis
    """
    significance_level: float = Field(..., description="Significance level (alpha)")
    min_proportion: float = Field(..., description="Minimum proportion of successes")
    num_of_samples: int = Field(..., description="Number of samples to collect")
    
    def evaluate(self, scores: List[float]) -> dict:
        """
        Evaluate if the proportion of successes meets the minimum threshold.
        
        Args:
            scores: List of scores (1-10) from Judge LLM
            
        Returns:
            Dictionary with evaluation results including pass/fail and statistics
        """
        # TODO: Implement one-sided proportion hypothesis test
        # 1. Convert scores to binary (6+ = success, 5 or lower = failure)
        # 2. Calculate observed proportion
        # 3. Perform one-sided hypothesis test
        # 4. Return pass/fail with statistical details
        return {}


class RatingMeanEvaluator(BaseModel):
    """
    Performs one-sided t-test to determine if mean rating exceeds baseline.
    
    Useful for subjective qualities like helpfulness or tone.
    
    Attributes:
        significance_level: Significance level for hypothesis test (e.g., 0.05)
        min_mean_rating: Minimum acceptable mean rating (e.g., 7.0)
        num_of_samples: Number of samples to collect for statistical analysis
    """
    significance_level: float = Field(..., description="Significance level (alpha)")
    min_mean_rating: float = Field(..., description="Minimum mean rating threshold")
    num_of_samples: int = Field(..., description="Number of samples to collect")
    
    def evaluate(self, scores: List[float]) -> dict:
        """
        Evaluate if the mean rating meets the minimum threshold.
        
        Args:
            scores: List of scores (1-10) from Judge LLM
            
        Returns:
            Dictionary with evaluation results including pass/fail and statistics
        """
        # TODO: Implement one-sided t-test
        # 1. Calculate sample mean and standard deviation
        # 2. Perform one-sided t-test against min_mean_rating
        # 3. Return pass/fail with statistical details
        return {}


class RatingProportionEvaluator(BaseModel):
    """
    Tests if proportion of ratings at or above threshold exceeds minimum.
    
    Similar to SuccessRateEvaluator but with configurable rating threshold.
    
    Attributes:
        significance_level: Significance level for hypothesis test (e.g., 0.05)
        min_rating: Minimum acceptable rating on 1-10 scale (e.g., 8)
        min_proportion: Minimum proportion of responses meeting min_rating (e.g., 0.75)
        num_of_samples: Number of samples to collect for statistical analysis
    """
    significance_level: float = Field(..., description="Significance level (alpha)")
    min_rating: int = Field(..., description="Minimum acceptable rating (1-10)")
    min_proportion: float = Field(..., description="Minimum proportion at or above min_rating")
    num_of_samples: int = Field(..., description="Number of samples to collect")
    
    def evaluate(self, scores: List[float]) -> dict:
        """
        Evaluate if the proportion of ratings meeting threshold is sufficient.
        
        Args:
            scores: List of scores (1-10) from Judge LLM
            
        Returns:
            Dictionary with evaluation results including pass/fail and statistics
        """
        # TODO: Implement one-sided proportion hypothesis test
        # 1. Convert scores to binary (>= min_rating = success, < min_rating = failure)
        # 2. Calculate observed proportion
        # 3. Perform one-sided hypothesis test
        # 4. Return pass/fail with statistical details
        return {}

