from pathlib import Path

import numpy as np
import tensorflow as tf

from prosthetic_rl_env import ProstheticRLEnv


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results" / "ddpg"

BASELINE_COST = 0.2602856207528844
EVALUATION_STEPS = 10


def report(label, result):
    cost = result["cost"]

    improvement = (
        (BASELINE_COST - cost)
        / BASELINE_COST * 100
    )

    print(f"\n--- {label} ---")
    print(f"Cost: {cost:.9f}")
    print(f"Improvement vs PSO: {improvement:.4f}%")
    print(f"Knee RMSE: {result['knee_rmse']:.4f} deg")
    print(f"Ankle RMSE: {result['ankle_rmse']:.4f} deg")


def main():

    # Load trained actor.
    actor = tf.keras.models.load_model(
        RESULTS_DIR / "actor.keras",
        compile=False
    )

    # Load best controller found during exploration.
    best_parameters = np.load(
        RESULTS_DIR / "best_parameters.npy"
    )

    env = ProstheticRLEnv(
        max_steps=EVALUATION_STEPS,
        current_limit=40.0
    )

    try:
        # --------------------------------------
        # Test 1: Reproduce PSO baseline
        # --------------------------------------

        state, info = env.reset(seed=42)

        baseline_result = env.last_result.copy()

        report("PSO BASELINE", baseline_result)

        # --------------------------------------
        # Test 2: Re-evaluate saved best action
        # --------------------------------------

        best_result = env.evaluate(
            best_parameters
        )

        report(
            "BEST DISCOVERED CONTROLLER",
            best_result
        )

        print("\nBest controller parameters:")
        print(best_parameters)

        # --------------------------------------
        # Test 3: Evaluate trained actor
        # without exploration noise
        # --------------------------------------

        state, info = env.reset(seed=42)

        trajectory = []

        for step in range(EVALUATION_STEPS):

            action = actor(
                np.asarray(
                    state, dtype=np.float32
                )[None, :],
                training=False
            ).numpy()[0]

            action = np.clip(action, -1, 1)

            (
                state,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(action)

            trajectory.append(info["cost"])

            print(
                f"Actor step {step + 1:02d}: "
                f"Cost = {info['cost']:.9f}"
            )

            if terminated or truncated:
                break

        report(
            "TRAINED ACTOR - FINAL CONTROLLER",
            info
        )

        print("\nFinal actor parameters:")
        print(env.parameters)

        print("\nBest actor trajectory cost:")
        print(f"{min(trajectory):.9f}")

        # --------------------------------------
        # Regression checks
        # --------------------------------------

        assert abs(
            baseline_result["cost"] - BASELINE_COST
        ) < 1e-4

        assert np.isfinite(best_result["cost"])
        assert np.all(np.isfinite(trajectory))

        print("\nDDPG VALIDATION COMPLETE.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
