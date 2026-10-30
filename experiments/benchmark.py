"""Parameter counts and single-sample inference latency.

The paper reports a ~1.5M-parameter model and sub-100 ms detection latency
(Table 8). This measures the reconstruction: parameter counts for the full
model and the ablation variants, and the median single-sample predict time
after warmup, on whatever machine runs it.

Usage:
    python -m experiments.benchmark
"""

import time

import numpy as np

import model as M
from experiments.common import count_params, save_result

WARMUP = 10
RUNS = 30


def latency_ms(model, timesteps, features):
    x = np.random.randn(1, timesteps, features).astype("float32")
    for _ in range(WARMUP):
        model.predict(x, verbose=0)
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        model.predict(x, verbose=0)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def main():
    from experiments.ablations import ARCH_VARIANTS
    report = {}
    for name, builder in ARCH_VARIANTS.items():
        m = builder(10, 6)
        report[name] = count_params(m)
        if name == "full":
            report[name]["median_latency_ms_batch1"] = latency_ms(m, 10, 6)
    report["paper_reports"] = ("~1.5M params, <100 ms latency (Table 8). "
                               "The full reconstruction here is 686,273 "
                               "trainable + 384 non-trainable = 686,657 "
                               "total, so it is not the published network.")
    for k, v in report.items():
        print(k, v)
    save_result("benchmark", report)


if __name__ == "__main__":
    main()
