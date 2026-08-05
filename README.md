# QxBin Use-Case Implementations

**Two concrete ideas. Two tiers. Real workloads.**

By Rupesh Malpani | pikk.company | QxBin Framework

Room-temperature probabilistic qubit simulation via Binary Probability Matrices, fractional states (biasⁿ / (1-bias)ᵐ), grids, and chains — running on ordinary classical hardware.

This repository ships **two focused, production-ready extensions** of the core QxBin logic:

| Tier | File | Use Case |
|------|------|----------|
| **Edge / Device** | `qxbin_edge_sensor_fusion.py` | Multi-sensor adaptive decision cubit (temp + vibration + battery → throttle / alert / sleep) |
| **Cloud / Server** | `qxbin_cloud_portfolio_risk.py` | Parallel Monte-Carlo portfolio / forecast stress-testing with VaR-style metrics |

Both preserve the exact QxBin mathematical primitives while targeting different computing envelopes and different real-world problems.

---

## 1. Edge Tier — Sensor Fusion Decision Cubit

**Goal**: Give a microcontroller or gateway a lightweight “personal cubit” that can fuse noisy sensors and collapse to an actionable decision without any cloud round-trip.

**Key features**
- Pure NumPy (easy to strip to pure Python if needed)
- Tiny memory footprint
- Explicit sensor → bias → fractional exponents → probability matrix → collapse pipeline
- Human-readable actions: `THROTTLE`, `ALERT`, `MONITOR`, `SLEEP`

**Run**
```bash
python qxbin_edge_sensor_fusion.py
```

Ideal for Pikk edge nodes, IoT gateways, robotics, battery-constrained devices.

---

## 2. Cloud / Server Tier — Portfolio Risk Engine

**Goal**: Run dozens to hundreds of independent QxBin probability matrices in parallel, evolve them under market-like bias + volatility, and extract distributional risk metrics.

**Key features**
- Numba parallel evolution (`prange`)
- Configurable scenario count & grid size
- Multi-step stress test with early stopping
- VaR 5 %, VaR 1 %, Expected Shortfall
- Extreme-scenario snapshot for inspection

**Run**
```bash
python qxbin_cloud_portfolio_risk.py
```

Ideal for research notebooks, batch risk jobs, energy-load forecasting, cash-flow scenario engines.

---

## Core QxBin Math (unchanged)

- Fractional states: `bias**n` and `(1-bias)**m`
- Binary Probability Matrix (2-D grid)
- Superposition-style blend
- Probabilistic measurement / collapse

No cryogenics. No massive labs. Democratizing quantum-inspired logic for everyone.

---

## License

This repository uses the **official default QxBin custom MIT license**.

- Free for testing, experimentation, internal organizational use, and building your own software.
- 51 % revenue share applies when you create and sell a commercial tool / product / API.
- Enterprise deployments and strategic partnerships are negotiable — reach out to [@rupeshmalpani](https://x.com/rupeshmalpani).

See the `LICENSE` file for full terms.

---

Part of the pikk-qxbin vision: **Democratizing advanced compute. Ship fast. Align incentives.**

X: [@rupeshmalpani](https://x.com/rupeshmalpani)  
Core framework: https://github.com/pikk-qxbin/qxbin
