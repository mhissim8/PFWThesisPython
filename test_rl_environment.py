import numpy as np
from prosthetic_rl_env import ProstheticRLEnv

env = ProstheticRLEnv(max_steps=5)

try:
    observation, info = env.reset(seed=42)

    print("\nInitial cost:", info["cost"])
    print("Observation shape:", observation.shape)

    assert observation.shape == (12,)
    assert np.isclose(info["cost"], 0.260286, atol=1e-4)

    for step in range(5):
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        print(f"\nStep {step + 1}")
        print(f"Cost: {info['cost']:.6f}")
        print(f"Reward: {reward:.6f}")
        print(f"Knee RMSE: {info['knee_rmse']:.3f}")
        print(f"Ankle RMSE: {info['ankle_rmse']:.3f}")

        assert np.isfinite(reward)
        assert observation.shape == (12,)

        if terminated or truncated:
            break

    print("\nRL ENVIRONMENT TEST PASSED.")

finally:
    env.close()
