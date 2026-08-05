"""
QxBin Edge Tier — Sensor Fusion Decision Cubit
==============================================
Room-temperature probabilistic logic for resource-constrained devices.

Use case: Adaptive decision under noisy multi-sensor input
(temperature, vibration, battery) on microcontrollers / edge nodes.

Core QxBin math preserved:
  - Binary Probability Matrix (grid)
  - Fractional exponents n, m for directed bias
  - Superposition blend + probabilistic collapse

Zero heavy dependencies. Pure NumPy (or pure Python fallback possible).
Designed for Pikk edge nodes, IoT gateways, personal experimentation.

By Rupesh Malpani | pikk.company | QxBin Framework
"""

import numpy as np
from typing import Dict, Tuple


class QxBinEdgeSensorFusion:
    """
    Lightweight personal cubit that fuses multiple noisy sensors
    into a single evolving probability matrix and collapses to
    actionable decisions (e.g. "throttle", "alert", "sleep").
    """

    def __init__(self, grid_size: int = 5):
        self.grid_size = grid_size
        # Start near-uniform — analog "unknown" state
        self.state = np.ones((grid_size, grid_size), dtype=np.float64)
        self.state /= self.state.sum()
        self.history = []

    def _normalize(self):
        s = self.state.sum()
        if s > 1e-12:
            self.state /= s
        else:
            self.state = np.ones_like(self.state) / self.state.size

    def fuse_sensors(
        self,
        temp_norm: float,
        vib_norm: float,
        battery_norm: float,
        n: int = 2,
        m: int = 1,
    ) -> np.ndarray:
        """
        Map three normalized sensors [0,1] into a directed
        QxBin superposition update.

        Higher temperature / vibration lean the bias one way,
        low battery leans the opposite. Exponents n,m control
        how sharply the probability matrix is steered.
        """
        # Composite bias from sensors (simple weighted sum, easy to tune)
        bias = 0.35 * temp_norm + 0.40 * vib_norm + 0.25 * (1.0 - battery_norm)
        bias = float(np.clip(bias, 0.05, 0.95))

        frac = bias ** n
        tail = (1.0 - bias) ** m

        # Coordinate vector → outer product builds the new matrix
        vec = np.linspace(frac, tail, self.grid_size)
        new_matrix = np.outer(vec, vec)

        # Superposition-style blend (coin still spinning)
        self.state = 0.55 * self.state + 0.45 * new_matrix
        self._normalize()

        self.history.append(
            {
                "bias": bias,
                "temp": temp_norm,
                "vib": vib_norm,
                "battery": battery_norm,
                "mean_prob": float(self.state.mean()),
            }
        )
        return self.state

    def measure(self) -> Tuple[str, float, np.ndarray]:
        """
        Collapse the probability matrix to a classical action.
        Returns (action_label, confidence, collapsed_matrix)
        """
        flat = self.state.flatten()
        idx = np.random.choice(len(flat), p=flat)
        collapsed = np.zeros_like(flat)
        collapsed[idx] = 1.0
        collapsed = collapsed.reshape(self.state.shape)

        # Map grid position → decision zones
        row, col = divmod(idx, self.grid_size)
        center = self.grid_size // 2

        if row < center and col < center:
            action = "THROTTLE"
        elif row >= center and col >= center:
            action = "ALERT"
        elif row < center:
            action = "MONITOR"
        else:
            action = "SLEEP"

        confidence = float(flat[idx])
        return action, confidence, collapsed

    def decide(
        self,
        sensors: Dict[str, float],
        n: int = 2,
        m: int = 1,
    ) -> Dict:
        """
        One-shot pipeline: fuse → measure → return human-readable decision.
        sensors keys: 'temp', 'vib', 'battery' (all expected in [0,1])
        """
        self.fuse_sensors(
            temp_norm=sensors.get("temp", 0.5),
            vib_norm=sensors.get("vib", 0.5),
            battery_norm=sensors.get("battery", 0.7),
            n=n,
            m=m,
        )
        action, conf, _ = self.measure()
        return {
            "action": action,
            "confidence": round(conf, 4),
            "state_mean": round(float(self.state.mean()), 4),
            "sensors": sensors,
        }


# ------------------------------------------------------------------
# Demo — realistic edge scenario
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("QxBin Edge — Sensor Fusion Decision Cubit")
    print("=" * 52)

    qx = QxBinEdgeSensorFusion(grid_size=5)

    scenarios = [
        {"temp": 0.82, "vib": 0.71, "battery": 0.45},  # hot + vibrating + low batt
        {"temp": 0.35, "vib": 0.22, "battery": 0.88},  # cool + quiet + full
        {"temp": 0.65, "vib": 0.55, "battery": 0.60},  # mixed
    ]

    for i, s in enumerate(scenarios, 1):
        decision = qx.decide(s, n=3, m=1)
        print(f"\nScenario {i}: {s}")
        print(f"  → Action      : {decision['action']}")
        print(f"  → Confidence  : {decision['confidence']}")
        print(f"  → State mean  : {decision['state_mean']}")

    print("\n✅ Edge cubit ready for microcontroller / gateway deployment.")
    print("   No cryogenics. No cloud. Just QxBin fractional logic.")
