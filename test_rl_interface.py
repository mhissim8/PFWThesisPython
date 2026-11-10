from pathlib import Path

import matlab
import matlab.engine

PROJECT_DIR = Path(__file__).resolve().parent

BASELINE_GAINS = [
    100.0,
    39.734139250294206,
    4.615117437656337,
    8.89292745637548,
    24.08102401285985,
    2.8065164211432663
]

CURRENT_LIMIT = 40.0

eng = matlab.engine.start_matlab()

try:
    eng.cd(str(PROJECT_DIR), nargout=0)

    def evaluate(name, alpha_k, alpha_a):
        action = BASELINE_GAINS + [alpha_k, alpha_a]

        result = eng.evaluateRLController(
            matlab.double(action),
            CURRENT_LIMIT,
            nargout=1
        )

        print(f"\n--- {name} ---")
        print(f"Cost: {float(result['cost']):.6f}")
        print(f"Reward: {float(result['reward']):.6f}")
        print(
            f"Knee RMSE: "
            f"{float(result['kneeRMSE']):.4f} deg"
        )
        print(
            f"Ankle RMSE: "
            f"{float(result['ankleRMSE']):.4f} deg"
        )
        print(
            f"Peak knee current: "
            f"{float(result['peakKneeCurrent']):.2f} A"
        )

        return result

    # Test 1: Original PID-only controller
    baseline = evaluate(
        "PSO BASELINE",
        0.0,
        0.0
    )

    # Test 2: Shared impedance scaling
    shared = evaluate(
        "SHARED IMPEDANCE",
        0.5,
        0.5
    )

    # Test 3: Independent impedance scaling
    independent = evaluate(
        "INDEPENDENT IMPEDANCE",
        0.25,
        0.75
    )

    # Confirm the previously validated PID tracking
    assert abs(
        float(baseline["kneeRMSE"]) - 28.8900
    ) < 0.01

    assert abs(
        float(baseline["ankleRMSE"]) - 4.4093
    ) < 0.01

    # Confirm all three simulations are stable
    for result in [baseline, shared, independent]:
        assert bool(result["stable"])

    # Reward should equal negative cost
    for result in [baseline, shared, independent]:
        assert abs(
            float(result["reward"]) +
            float(result["cost"])
        ) < 1e-9

    print("\nALL RL INTERFACE TESTS PASSED.")

finally:
    eng.quit()
