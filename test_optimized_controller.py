from pathlib import Path
import matlab
import matlab.engine

PROJECT_DIR = Path(__file__).resolve().parent

print("Starting MATLAB Engine...")
eng = matlab.engine.start_matlab()

try:
    eng.cd(str(PROJECT_DIR), nargout=0)

    # Load previously optimized PID gains.
    eng.eval(
        "load('results/optimizedCurrentLimits.mat');",
        nargout=0
    )

    # Identify the row corresponding to 40 A.
    eng.eval(
        "idx = find(currentLimits == 40, 1);",
        nargout=0
    )

    # Extract the six optimized gains.
    gains = eng.eval(
        "bestGainMatrix(idx,:)",
        nargout=1
    )

    print("Optimized gains:", gains)

    # Evaluate PID-only operation.
    result = eng.evaluateFromPython(
        gains,
        0.0,
        40.0,
        nargout=1
    )

    print("\n--- OPTIMIZED 40 A CONTROLLER ---")
    print(f"Knee RMSE: {float(result['kneeRMSE']):.4f} deg")
    print(f"Ankle RMSE: {float(result['ankleRMSE']):.4f} deg")
    print(f"Interface cost: {float(result['cost']):.6f}")

    print(
        "Peak knee current:",
        float(result["peakKneeCurrent"]),
        "A"
    )

    print(
        "Peak ankle current:",
        float(result["peakAnkleCurrent"]),
        "A"
    )

    print(
        "Knee current limiting:",
        float(result["kneeCurrentLimitedPct"]),
        "%"
    )

finally:
    eng.quit()
