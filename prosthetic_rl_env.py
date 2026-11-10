from pathlib import Path

import gymnasium as gym
import matlab
import matlab.engine
import numpy as np
from gymnasium import spaces


class ProstheticRLEnv(gym.Env):
    """
    Sequential PID and impedance parameter tuning.

    One step = one complete MATLAB gait simulation.
    No physical motor commands are generated.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps=10, current_limit=40.0):
        super().__init__()

        self.project_dir = Path(__file__).resolve().parent
        self.current_limit = float(current_limit)
        self.max_steps = int(max_steps)

        self.lower = np.zeros(8, dtype=np.float64)

        self.upper = np.array(
            [100, 40, 10, 100, 40, 10, 1, 1],
            dtype=np.float64
        )

        self.baseline = np.array([
            100.0,
            39.734139250294206,
            4.615117437656337,
            8.89292745637548,
            24.08102401285985,
            2.8065164211432663,
            0.0,
            0.0
        ], dtype=np.float64)

        # Actions represent normalized parameter changes.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(8,),
            dtype=np.float32
        )

        # Eight normalized controller parameters,
        # plus cost, knee RMSE, ankle RMSE,
        # and knee current-limiting fraction.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32
        )

        print("Starting MATLAB Engine...")
        self.eng = matlab.engine.start_matlab()

        self.eng.cd(
            str(self.project_dir),
            nargout=0
        )

        self.parameters = self.baseline.copy()
        self.previous_cost = None
        self.steps = 0
        self.last_result = None

    def evaluate(self, parameters):
        result = self.eng.evaluateRLController(
            matlab.double(parameters.tolist()),
            self.current_limit,
            nargout=1
        )

        return {
            "cost": float(result["cost"]),
            "knee_rmse": float(result["kneeRMSE"]),
            "ankle_rmse": float(result["ankleRMSE"]),
            "knee_limit": float(
                result.get("kneeCurrentLimitedPct", 0.0)
            ),
            "stable": bool(result["stable"])
        }

    def observation(self, result):
        normalized_parameters = (
            self.parameters / self.upper
        )

        metrics = np.array([
            result["cost"],
            result["knee_rmse"] / 90.0,
            result["ankle_rmse"] / 90.0,
            result["knee_limit"] / 100.0
        ])

        return np.concatenate([
            normalized_parameters,
            metrics
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.parameters = self.baseline.copy()
        self.steps = 0

        result = self.evaluate(self.parameters)

        self.previous_cost = result["cost"]
        self.last_result = result

        return self.observation(result), {
            "cost": result["cost"]
        }

    def step(self, action):
        action = np.asarray(
            action, dtype=np.float64
        ).reshape(8)

        if not np.all(np.isfinite(action)):
            raise ValueError("Action contains nonfinite values.")

        action = np.clip(action, -1.0, 1.0)

        # Maximum change per learning step:
        # 5% of each parameter's full range.
        delta = 0.05 * self.upper * action

        self.parameters = np.clip(
            self.parameters + delta,
            self.lower,
            self.upper
        )

        result = self.evaluate(self.parameters)

        # Positive reward for improvement.
        reward = 100.0 * (
            self.previous_cost - result["cost"]
        )

        self.previous_cost = result["cost"]
        self.last_result = result

        self.steps += 1

        terminated = not result["stable"]
        truncated = self.steps >= self.max_steps

        if terminated:
            reward = -100.0

        info = {
            **result,
            "parameters": self.parameters.copy()
        }

        return (
            self.observation(result),
            float(reward),
            terminated,
            truncated,
            info
        )

    def close(self):
        if getattr(self, "eng", None) is not None:
            self.eng.quit()
            self.eng = None
