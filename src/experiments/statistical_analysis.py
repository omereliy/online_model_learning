"""Statistical analysis module for comparing algorithm performance."""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from scipy import stats


@dataclass
class StatisticalResult:
    """Stores results from statistical comparison of two algorithms."""

    algorithm1_name: str
    algorithm2_name: str
    metric_name: str
    algorithm1_mean: float
    algorithm1_std: float
    algorithm1_ci: Tuple[float, float]
    algorithm2_mean: float
    algorithm2_std: float
    algorithm2_ci: Tuple[float, float]
    t_statistic: float
    p_value: float
    effect_size: float  # Cohen's d
    significant: bool  # p < alpha
    interpretation: str


class StatisticalAnalyzer:
    """Provides statistical analysis methods for algorithm comparison."""

    def compare_algorithms(
        self,
        results1: List[float],
        results2: List[float],
        algorithm1_name: str,
        algorithm2_name: str,
        metric_name: str,
        alpha: float = 0.05,
        confidence: float = 0.95
    ) -> StatisticalResult:
        """Compare two algorithms using paired t-test and effect size.

        Args:
            results1: Performance results from algorithm 1 (one value per trial)
            results2: Performance results from algorithm 2 (one value per trial)
            algorithm1_name: Name of first algorithm
            algorithm2_name: Name of second algorithm
            metric_name: Name of the metric being compared
            alpha: Significance level (default 0.05)
            confidence: Confidence level for intervals (default 0.95)

        Returns:
            StatisticalResult with complete statistical comparison
        """
        # Convert to numpy arrays
        data1 = np.array(results1, dtype=float)
        data2 = np.array(results2, dtype=float)

        # Ensure same length for paired comparison
        if len(data1) != len(data2):
            raise ValueError(f"Results must have same length for paired comparison: {len(data1)} vs {len(data2)}")

        # Calculate basic statistics (convert to Python floats to avoid numpy scalar issues)
        mean1 = float(np.mean(data1))
        std1 = float(np.std(data1, ddof=1)) if len(data1) > 1 else 0.0
        mean2 = float(np.mean(data2))
        std2 = float(np.std(data2, ddof=1)) if len(data2) > 1 else 0.0

        # Calculate confidence intervals
        ci1 = self._compute_confidence_interval(data1, confidence)
        ci2 = self._compute_confidence_interval(data2, confidence)

        # Perform paired t-test
        if len(data1) > 1:
            # Check if all differences are the same (no variance)
            differences = data1 - data2
            if np.std(differences) == 0:
                # All differences are identical
                if np.mean(differences) == 0:
                    # Identical data
                    t_stat = 0.0
                    p_value = 1.0
                else:
                    # Constant non-zero difference
                    t_stat = float('inf') * np.sign(np.mean(differences))
                    p_value = 0.0
            else:
                # Use paired t-test for related samples
                t_stat, p_value = stats.ttest_rel(data1, data2)
                # Convert numpy scalars to Python floats
                t_stat = float(t_stat)
                p_value = float(p_value)
        else:
            # Single observation - can't compute t-test
            t_stat = float('inf') if mean1 != mean2 else 0.0
            p_value = 0.0 if mean1 != mean2 else 1.0

        # Calculate effect size (Cohen's d)
        effect_size = float(self._compute_cohens_d(data1, data2))

        # Determine significance (convert to Python bool to avoid numpy bool issues)
        significant = bool(p_value < alpha)

        # Create result object (interpretation will be set after)
        result = StatisticalResult(
            algorithm1_name=algorithm1_name,
            algorithm2_name=algorithm2_name,
            metric_name=metric_name,
            algorithm1_mean=mean1,
            algorithm1_std=std1,
            algorithm1_ci=ci1,
            algorithm2_mean=mean2,
            algorithm2_std=std2,
            algorithm2_ci=ci2,
            t_statistic=t_stat,
            p_value=p_value,
            effect_size=effect_size,
            significant=significant,
            interpretation=""  # Will be set next
        )

        # Generate interpretation
        result.interpretation = self._interpret_result(result)

        return result

    def _compute_cohens_d(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate Cohen's d effect size for two datasets.

        Cohen's d = (mean1 - mean2) / pooled_std

        Args:
            data1: First dataset
            data2: Second dataset

        Returns:
            Cohen's d effect size (negative if data2 has lower mean)
        """
        mean1 = np.mean(data1)
        mean2 = np.mean(data2)

        # Calculate pooled standard deviation
        n1, n2 = len(data1), len(data2)

        if n1 <= 1 and n2 <= 1:
            # Can't compute std with single values
            return 0.0 if mean1 == mean2 else float('inf') * np.sign(mean1 - mean2)

        # Use pooled standard deviation for independent samples
        # For paired samples, could use std of differences, but Cohen's d traditionally uses pooled std
        var1 = np.var(data1, ddof=1) if n1 > 1 else 0.0
        var2 = np.var(data2, ddof=1) if n2 > 1 else 0.0

        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2) if (n1 + n2) > 2 else max(var1, var2)
        pooled_std = np.sqrt(pooled_var) if pooled_var > 0 else 1.0

        if pooled_std == 0:
            # Both have same values
            return 0.0 if mean1 == mean2 else float('inf') * np.sign(mean1 - mean2)

        return (mean1 - mean2) / pooled_std

    def _compute_confidence_interval(
        self,
        data: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for a dataset.

        Args:
            data: Dataset
            confidence: Confidence level (e.g., 0.95 for 95% CI)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        mean = float(np.mean(data))
        n = len(data)

        if n == 1:
            # Single value - no interval
            return (mean, mean)

        # Calculate standard error
        std_err = stats.sem(data)

        if std_err == 0:
            # All values are the same
            return (mean, mean)

        # Calculate confidence interval using t-distribution
        # (more appropriate than normal for small samples)
        margin = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)

        return (float(mean - margin), float(mean + margin))

    def _bonferroni_correction(
        self,
        p_values: List[float],
        alpha: float = 0.05
    ) -> List[bool]:
        """Apply Bonferroni correction for multiple comparisons.

        Args:
            p_values: List of p-values from multiple tests
            alpha: Overall significance level

        Returns:
            List of booleans indicating which tests are significant after correction
        """
        n_tests = len(p_values)
        if n_tests == 0:
            return []

        # Bonferroni correction: divide alpha by number of tests
        corrected_alpha = alpha / n_tests

        return [p < corrected_alpha for p in p_values]

    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size.

        Args:
            d: Cohen's d value (absolute value)

        Returns:
            String interpretation: "small", "medium", or "large"
        """
        abs_d = abs(d)

        if abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def _interpret_result(self, result: StatisticalResult) -> str:
        """Generate human-readable interpretation of statistical result.

        Args:
            result: StatisticalResult object

        Returns:
            String interpretation of the results
        """
        # Determine which algorithm is better (if any)
        if not result.significant:
            return (f"No significant difference between {result.algorithm1_name} "
                   f"and {result.algorithm2_name} for {result.metric_name} "
                   f"(p = {result.p_value:.3f})")

        # Determine which performed better based on mean values
        # For metrics like sample_complexity, runtime, errors: lower is better
        # For metrics like accuracy, success_rate: higher is better
        # We'll assume lower is better by default (can be parameterized later)

        # Check if algorithm2 has lower mean (better for minimization metrics)
        if result.algorithm2_mean < result.algorithm1_mean:
            better_algo = result.algorithm2_name
            worse_algo = result.algorithm1_name
            mean_diff = result.algorithm1_mean - result.algorithm2_mean
        else:
            better_algo = result.algorithm1_name
            worse_algo = result.algorithm2_name
            mean_diff = result.algorithm2_mean - result.algorithm1_mean

        effect_interpretation = self._interpret_effect_size(result.effect_size)

        return (f"{better_algo} performs significantly better than {worse_algo} "
               f"for {result.metric_name} (p = {result.p_value:.3f}, "
               f"{effect_interpretation} effect size, d = {abs(result.effect_size):.2f}). "
               f"Mean difference: {abs(mean_diff):.2f}")