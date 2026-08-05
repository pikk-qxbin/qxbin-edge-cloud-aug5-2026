"""
QxBin Cloud / Server Tier — Parallel Monte-Carlo Portfolio Risk Engine
======================================================================
Scalable ensemble of Binary Probability Matrices for uncertainty modeling.

Use case: Rapid scenario stress-testing of portfolios, cash-flow forecasts,
or energy-load curves under correlated uncertainty.

Core QxBin math preserved + accelerated:
  - Ensemble of independent probability grids
  - Fractional exponents n, m for directed contributions
  - Parallel evolution (Numba)
  - Aggregate statistics + risk metrics (VaR-style)

Ideal for batch jobs, optimization loops, research notebooks,
and high-speed cloud workloads on classical hardware.

By Rupesh Malpani | pikk.company | QxBin Framework
"""

import numpy as np
from numba import njit, prange
from typing import Dict, List, Optional


@njit(parallel=True, fastmath=True)
def _evolve_ensemble(states, biases, ns, ms):
    """Numba-parallel QxBin evolution of many cubit matrices.
    Uses outer-product directed contribution (same spirit as Edge tier)
    so each scenario keeps distinct structure under different n/m/bias.
    """
    n_cubits = states.shape[0]
    grid = states.shape[1]
    for i in prange(n_cubits):
        b = biases[i]
        nn = ns[i]
        mm = ms[i]

        frac = b ** nn
        tail = (1.0 - b) ** mm

        # Build directed coordinate vector → outer product
        # (Numba-friendly loop instead of np.linspace/outer)
        vec = np.empty(grid, dtype=np.float64)
        if grid > 1:
            step = (tail - frac) / (grid - 1)
            for k in range(grid):
                vec[k] = frac + k * step
        else:
            vec[0] = frac

        new_matrix = np.empty((grid, grid), dtype=np.float64)
        for r in range(grid):
            for c in range(grid):
                new_matrix[r, c] = vec[r] * vec[c]

        # Superposition-style blend
        blended = 0.55 * states[i] + 0.45 * new_matrix

        total = blended.sum()
        if total > 1e-12:
            states[i] = blended / total
        else:
            states[i] = np.ones_like(blended) / blended.size
    return states


class QxBinCloudPortfolioRisk:
    """
    Ensemble of QxBin cubits representing alternative future paths
    of a portfolio (or any scalar process). Each cubit is a full
    probability matrix; the ensemble yields distributional risk metrics.
    """

    def __init__(self, num_scenarios: int = 64, grid_size: int = 6):
        self.num_scenarios = num_scenarios
        self.grid_size = grid_size
        self.states = np.random.rand(num_scenarios, grid_size, grid_size).astype(np.float64)
        # Normalize each matrix
        for i in range(num_scenarios):
            s = self.states[i].sum()
            if s > 0:
                self.states[i] /= s

    def evolve(
        self,
        market_bias: float = 0.62,
        volatility: float = 0.18,
        n_range: tuple = (1, 4),
        m_range: tuple = (1, 4),
    ) -> np.ndarray:
        """
        Evolve the entire ensemble one step.
        market_bias controls overall directional lean;
        volatility spreads the individual scenario biases.
        """
        biases = np.clip(
            np.random.normal(market_bias, volatility, self.num_scenarios),
            0.05,
            0.95,
        )
        ns = np.random.randint(n_range[0], n_range[1] + 1, self.num_scenarios)
        ms = np.random.randint(m_range[0], m_range[1] + 1, self.num_scenarios)

        self.states = _evolve_ensemble(self.states, biases, ns, ms)
        return self.states.mean(axis=0)

    def _scenario_scores(self) -> np.ndarray:
        """
        Project each probability matrix to a scalar 'value' score.
        Higher mass in the top-right quadrant → bullish score.
        This gives meaningful variation across scenarios.
        """
        g = self.grid_size
        half = g // 2
        scores = np.zeros(self.num_scenarios, dtype=np.float64)
        for i in range(self.num_scenarios):
            # Concentration in high-value zone + entropy-like spread
            high_zone = self.states[i, half:, half:].sum()
            low_zone = self.states[i, :half, :half].sum()
            scores[i] = high_zone - low_zone   # signed directional score
        return scores

    def run_stress_test(
        self,
        steps: int = 40,
        target_mean: Optional[float] = None,
        market_bias: float = 0.60,
        volatility: float = 0.20,
    ) -> Dict:
        """
        Multi-step Monte-Carlo stress test.
        Returns summary risk metrics derived from the final ensemble scores.
        """
        scores_over_time = []
        for step in range(steps):
            self.evolve(market_bias=market_bias, volatility=volatility)
            scores = self._scenario_scores()
            scores_over_time.append(float(scores.mean()))

            if target_mean is not None and abs(scores_over_time[-1] - target_mean) < 0.004:
                break

        final_scores = self._scenario_scores()

        # VaR-style metrics on the signed scores (lower tail = downside risk)
        var_5 = float(np.percentile(final_scores, 5))
        var_1 = float(np.percentile(final_scores, 1))
        expected_shortfall = float(final_scores[final_scores <= var_5].mean()) if np.any(final_scores <= var_5) else var_5

        return {
            "steps_run": len(scores_over_time),
            "final_ensemble_score_mean": round(float(final_scores.mean()), 5),
            "final_ensemble_score_std": round(float(final_scores.std()), 5),
            "VaR_5pct": round(var_5, 5),
            "VaR_1pct": round(var_1, 5),
            "Expected_Shortfall_5pct": round(expected_shortfall, 5),
            "score_trajectory": [round(m, 5) for m in scores_over_time[:: max(1, len(scores_over_time)//8)]],
        }

    def scenario_snapshot(self, top_k: int = 5) -> List[Dict]:
        """Return the k highest- and lowest-score scenarios for inspection."""
        scores = self._scenario_scores()
        order = np.argsort(scores)
        extremes = []
        for idx in list(order[:top_k]) + list(order[-top_k:]):
            extremes.append(
                {
                    "scenario_id": int(idx),
                    "score": round(float(scores[idx]), 5),
                    "label": "tail" if scores[idx] < scores.mean() else "bull",
                }
            )
        return extremes


# ------------------------------------------------------------------
# Demo — portfolio stress under elevated volatility
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("QxBin Cloud — Parallel Portfolio Risk Engine")
    print("=" * 55)

    engine = QxBinCloudPortfolioRisk(num_scenarios=96, grid_size=6)
    print(f"Initialized {engine.num_scenarios} parallel cubit scenarios "
          f"({engine.grid_size}×{engine.grid_size} grids)")

    print("\nRunning stress test (elevated volatility)...")
    results = engine.run_stress_test(
        steps=50,
        market_bias=0.58,
        volatility=0.25,
    )

    print("\nRisk Summary")
    print("-" * 40)
    for k, v in results.items():
        if k != "score_trajectory":
            print(f"  {k:28s}: {v}")

    print("\nSample score trajectory (subsampled):")
    print(" ", results["score_trajectory"])

    print("\nExtreme scenarios:")
    for s in engine.scenario_snapshot(top_k=3):
        print(f"  id={s['scenario_id']:3d}  score={s['score']:.5f}  [{s['label']}]")

    print("\n✅ Cloud ensemble ready. Same QxBin fractional logic,")
    print("   now at scale for risk, forecasting, and optimization.")
