"""Tests for statistical analysis module."""

import pytest
import numpy as np
from typing import List
from dataclasses import dataclass

from src.experiments.statistical_analysis import (
    StatisticalResult,
    StatisticalAnalyzer
)


class TestStatisticalResult:
    """Test StatisticalResult dataclass."""

    def test_dataclass_fields(self):
        """Test that StatisticalResult contains all required fields."""
        result = StatisticalResult(
            algorithm1_name="OLAM",
            algorithm2_name="InfoGain",
            metric_name="sample_complexity",
            algorithm1_mean=50.0,
            algorithm1_std=10.0,
            algorithm1_ci=(45.0, 55.0),
            algorithm2_mean=45.0,
            algorithm2_std=8.0,
            algorithm2_ci=(41.0, 49.0),
            t_statistic=2.5,
            p_value=0.03,
            effect_size=0.55,
            significant=True,
            interpretation="Algorithm 2 performs significantly better with medium effect size"
        )

        assert result.algorithm1_name == "OLAM"
        assert result.algorithm2_name == "InfoGain"
        assert result.metric_name == "sample_complexity"
        assert result.algorithm1_mean == 50.0
        assert result.algorithm1_std == 10.0
        assert result.algorithm1_ci == (45.0, 55.0)
        assert result.algorithm2_mean == 45.0
        assert result.algorithm2_std == 8.0
        assert result.algorithm2_ci == (41.0, 49.0)
        assert result.t_statistic == 2.5
        assert result.p_value == 0.03
        assert result.effect_size == 0.55
        assert result.significant is True
        assert "medium effect size" in result.interpretation


class TestStatisticalAnalyzer:
    """Test StatisticalAnalyzer class."""

    def test_known_t_test_example(self):
        """Test paired t-test with known values.

        Example from statistics textbook:
        Data1: [10, 12, 14]
        Data2: [15, 17, 19]

        Expected results (hand-calculated):
        - Difference: [-5, -5, -5]
        - Mean difference: -5.0
        - t-statistic: undefined (std = 0) -> handle gracefully

        Modified example with variance:
        Data1: [10, 12, 14]
        Data2: [15, 18, 18]
        Differences: [-5, -6, -4], mean = -5.0, std = 1.0
        t = -5.0 / (1.0 / sqrt(3)) = -8.66
        """
        analyzer = StatisticalAnalyzer()

        # Test with variance
        data1 = [10, 12, 14]
        data2 = [15, 18, 18]

        result = analyzer.compare_algorithms(
            results1=data1,
            results2=data2,
            algorithm1_name="Algo1",
            algorithm2_name="Algo2",
            metric_name="test_metric"
        )

        # Check basic statistics
        assert result.algorithm1_mean == pytest.approx(12.0, rel=1e-3)
        assert result.algorithm2_mean == pytest.approx(17.0, rel=1e-3)

        # Check t-statistic (paired t-test)
        # t = mean_diff / (std_diff / sqrt(n))
        # differences: [-5, -6, -4], mean = -5.0, std = 1.0
        # t = -5.0 / (1.0 / sqrt(3)) = -8.66
        assert result.t_statistic == pytest.approx(-8.66, rel=1e-1)
        assert result.p_value < 0.05  # Should be significant
        assert result.significant is True

    def test_cohens_d_calculation(self):
        """Test Cohen's d effect size calculation.

        Cohen's d = (mean1 - mean2) / pooled_std

        Example:
        Data1: [10, 12, 14], mean = 12, std = 2
        Data2: [20, 22, 24], mean = 22, std = 2
        Pooled std = 2 (when both stds are equal)
        Cohen's d = (12 - 22) / 2 = -5.0 (very large effect)
        """
        analyzer = StatisticalAnalyzer()

        data1 = [10, 12, 14]
        data2 = [20, 22, 24]

        result = analyzer.compare_algorithms(
            results1=data1,
            results2=data2,
            algorithm1_name="Algo1",
            algorithm2_name="Algo2",
            metric_name="test_metric"
        )

        # Cohen's d should be large negative (algo2 much better)
        assert result.effect_size == pytest.approx(-5.0, rel=1e-1)

        # Check interpretation
        interpretation = analyzer._interpret_effect_size(abs(result.effect_size))
        assert interpretation == "large"

    def test_confidence_interval(self):
        """Test confidence interval calculation.

        For normal distribution with known values:
        Data: [10, 12, 14, 16, 18]
        Mean = 14, Std = 3.16
        95% CI = mean ± 1.96 * (std / sqrt(n))
        CI = 14 ± 1.96 * (3.16 / sqrt(5))
        CI = 14 ± 2.77 = (11.23, 16.77)
        """
        analyzer = StatisticalAnalyzer()

        data1 = [10, 12, 14, 16, 18]
        data2 = [15, 15, 15, 15, 15]  # Constant for simplicity

        result = analyzer.compare_algorithms(
            results1=data1,
            results2=data2,
            algorithm1_name="Algo1",
            algorithm2_name="Algo2",
            metric_name="test_metric"
        )

        # Check CI for algorithm1
        lower, upper = result.algorithm1_ci
        assert lower == pytest.approx(11.0, abs=1.0)
        assert upper == pytest.approx(17.0, abs=1.0)

        # Check CI for algorithm2 (should be tight since no variance)
        lower2, upper2 = result.algorithm2_ci
        assert lower2 == pytest.approx(15.0, abs=0.1)
        assert upper2 == pytest.approx(15.0, abs=0.1)

    def test_bonferroni_correction(self):
        """Test Bonferroni correction for multiple comparisons."""
        analyzer = StatisticalAnalyzer()

        # Test with multiple p-values
        p_values = [0.01, 0.04, 0.06, 0.20]
        alpha = 0.05

        corrected_significant = analyzer._bonferroni_correction(p_values, alpha)

        # With Bonferroni correction: alpha_corrected = 0.05 / 4 = 0.0125
        # Only first p-value (0.01) should be significant
        assert corrected_significant == [True, False, False, False]

    def test_equal_performance(self):
        """Test edge case where algorithms perform equally."""
        analyzer = StatisticalAnalyzer()

        data1 = [10, 10, 10, 10, 10]
        data2 = [10, 10, 10, 10, 10]

        result = analyzer.compare_algorithms(
            results1=data1,
            results2=data2,
            algorithm1_name="Algo1",
            algorithm2_name="Algo2",
            metric_name="test_metric"
        )

        # Should have no difference
        assert result.algorithm1_mean == result.algorithm2_mean
        assert result.effect_size == pytest.approx(0.0, abs=1e-6)
        assert result.p_value == pytest.approx(1.0, abs=0.1)  # No significant difference
        assert result.significant is False
        assert "no significant difference" in result.interpretation.lower()

    def test_small_sample_size(self):
        """Test with small sample size (n=3)."""
        analyzer = StatisticalAnalyzer()

        # Small sample
        data1 = [10, 12, 14]
        data2 = [11, 13, 15]

        result = analyzer.compare_algorithms(
            results1=data1,
            results2=data2,
            algorithm1_name="Algo1",
            algorithm2_name="Algo2",
            metric_name="test_metric"
        )

        # Should handle gracefully despite small sample
        assert result.algorithm1_mean == pytest.approx(12.0, rel=1e-3)
        assert result.algorithm2_mean == pytest.approx(13.0, rel=1e-3)
        assert result.p_value is not None
        assert result.effect_size is not None

    def test_integration_with_experiment_results(self):
        """Test with realistic experiment results format."""
        analyzer = StatisticalAnalyzer()

        # Simulate multiple trial results (e.g., sample complexity across 10 trials)
        olam_results = [45, 52, 48, 50, 47, 51, 49, 53, 46, 49]
        info_gain_results = [38, 41, 39, 42, 40, 37, 43, 39, 41, 40]

        result = analyzer.compare_algorithms(
            results1=olam_results,
            results2=info_gain_results,
            algorithm1_name="OLAM",
            algorithm2_name="Information Gain",
            metric_name="sample_complexity"
        )

        # Information Gain should be significantly better (lower sample complexity)
        assert result.algorithm1_mean > result.algorithm2_mean
        assert result.p_value < 0.05
        assert result.significant is True
        assert result.effect_size > 0  # Positive because algo1 has higher mean (Cohen's d = (mean1 - mean2) / pooled_std)

        # Check interpretation mentions which is better (Info Gain has lower values = better)
        assert "Information Gain" in result.interpretation

    def test_interpretation_helpers(self):
        """Test effect size and result interpretation helpers."""
        analyzer = StatisticalAnalyzer()

        # Test effect size interpretations
        assert analyzer._interpret_effect_size(0.1) == "small"
        assert analyzer._interpret_effect_size(0.3) == "small"
        assert analyzer._interpret_effect_size(0.5) == "medium"
        assert analyzer._interpret_effect_size(0.7) == "medium"
        assert analyzer._interpret_effect_size(0.8) == "large"
        assert analyzer._interpret_effect_size(1.5) == "large"

        # Test result interpretation
        result = StatisticalResult(
            algorithm1_name="OLAM",
            algorithm2_name="InfoGain",
            metric_name="sample_complexity",
            algorithm1_mean=50.0,
            algorithm1_std=10.0,
            algorithm1_ci=(45.0, 55.0),
            algorithm2_mean=40.0,
            algorithm2_std=8.0,
            algorithm2_ci=(36.0, 44.0),
            t_statistic=3.5,
            p_value=0.001,
            effect_size=1.11,  # Large effect
            significant=True,
            interpretation=""
        )

        interpretation = analyzer._interpret_result(result)
        assert "significantly" in interpretation.lower()
        assert "large effect" in interpretation.lower()
        assert "InfoGain" in interpretation or "Algorithm 2" in interpretation