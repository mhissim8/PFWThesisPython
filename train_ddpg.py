from pathlib import Path
from collections import deque
import csv
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from prosthetic_rl_env import ProstheticRLEnv


# ==========================================
# CONFIGURATION
# ==========================================

SEED = 42
EPISODES = 30
STEPS_PER_EPISODE = 10

GAMMA = 0.99
TAU = 0.005

ACTOR_LR = 1e-4
CRITIC_LR = 1e-3

BATCH_SIZE = 64
REPLAY_CAPACITY = 10000
WARMUP_STEPS = 64

EXPLORATION_STD = 0.25

BASELINE_COST = 0.2602856207528844

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "ddpg"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==========================================
# ACTOR NETWORK
# ==========================================

def build_actor(state_dim, action_dim):
    inputs = keras.Input(shape=(state_dim,))

    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dense(128, activation="relu")(x)

    outputs = layers.Dense(
        action_dim,
        activation="tanh"
    )(x)

    return keras.Model(inputs, outputs)


# ==========================================
# CRITIC NETWORK
# ==========================================

def build_critic(state_dim, action_dim):
    state_input = keras.Input(shape=(state_dim,))
    action_input = keras.Input(shape=(action_dim,))

    x = layers.Concatenate()([
        state_input,
        action_input
    ])

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)

    q_value = layers.Dense(1)(x)

    return keras.Model(
        [state_input, action_input],
        q_value
    )


# ==========================================
# EXPERIENCE REPLAY
# ==========================================

class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward,
            next_state, done):

        self.buffer.append((
            np.asarray(state, dtype=np.float32).copy(),
            np.asarray(action, dtype=np.float32).copy(),
            float(reward),
            np.asarray(next_state, dtype=np.float32).copy(),
            float(done)
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = (
            zip(*batch)
        )

        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32)[:, None],
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32)[:, None]
        )

    def __len__(self):
        return len(self.buffer)


# ==========================================
# DDPG AGENT
# ==========================================

class DDPGAgent:

    def __init__(self, state_dim, action_dim):

        self.actor = build_actor(
            state_dim, action_dim
        )

        self.critic = build_critic(
            state_dim, action_dim
        )

        self.target_actor = build_actor(
            state_dim, action_dim
        )

        self.target_critic = build_critic(
            state_dim, action_dim
        )

        self.target_actor.set_weights(
            self.actor.get_weights()
        )

        self.target_critic.set_weights(
            self.critic.get_weights()
        )

        self.actor_optimizer = keras.optimizers.Adam(
            learning_rate=ACTOR_LR
        )

        self.critic_optimizer = keras.optimizers.Adam(
            learning_rate=CRITIC_LR
        )

        self.replay = ReplayBuffer(REPLAY_CAPACITY)

    def select_action(self, state, noise_std=0.0):
        state = np.asarray(
            state, dtype=np.float32
        )[None, :]

        action = self.actor(
            state, training=False
        ).numpy()[0]

        if noise_std > 0:
            action += np.random.normal(
                0.0, noise_std,
                size=action.shape
            )

        return np.clip(
            action, -1.0, 1.0
        ).astype(np.float32)

    def update_targets(self):

        for target, source in [
            (self.target_actor, self.actor),
            (self.target_critic, self.critic)
        ]:
            for target_var, source_var in zip(
                target.weights, source.weights
            ):
                target_var.assign(
                    TAU * source_var +
                    (1.0 - TAU) * target_var
                )

    def train_step(self):

        if len(self.replay) < max(
            BATCH_SIZE, WARMUP_STEPS
        ):
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = self.replay.sample(BATCH_SIZE)

        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards)
        next_states = tf.convert_to_tensor(next_states)
        dones = tf.convert_to_tensor(dones)

        # Critic update
        with tf.GradientTape() as tape:

            next_actions = self.target_actor(
                next_states, training=False
            )

            next_q = self.target_critic(
                [next_states, next_actions],
                training=False
            )

            targets = tf.stop_gradient(
                rewards +
                GAMMA * (1.0 - dones) * next_q
            )

            current_q = self.critic(
                [states, actions],
                training=True
            )

            critic_loss = tf.reduce_mean(
                tf.square(targets - current_q)
            )

        gradients = tape.gradient(
            critic_loss,
            self.critic.trainable_variables
        )

        self.critic_optimizer.apply_gradients(
            zip(
                gradients,
                self.critic.trainable_variables
            )
        )

        # Actor update
        with tf.GradientTape() as tape:

            predicted_actions = self.actor(
                states, training=True
            )

            predicted_q = self.critic(
                [states, predicted_actions],
                training=False
            )

            actor_loss = -tf.reduce_mean(
                predicted_q
            )

        gradients = tape.gradient(
            actor_loss,
            self.actor.trainable_variables
        )

        self.actor_optimizer.apply_gradients(
            zip(
                gradients,
                self.actor.trainable_variables
            )
        )

        self.update_targets()

        return (
            float(actor_loss.numpy()),
            float(critic_loss.numpy())
        )


# ==========================================
# TRAINING
# ==========================================

def main():

    env = ProstheticRLEnv(
        max_steps=STEPS_PER_EPISODE,
        current_limit=40.0
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = DDPGAgent(state_dim, action_dim)

    best_cost = BASELINE_COST
    best_parameters = env.baseline.copy()

    history = []

    try:

        for episode in range(EPISODES):

            state, info = env.reset(
                seed=SEED + episode
            )

            total_reward = 0.0
            actor_losses = []
            critic_losses = []

            for step in range(STEPS_PER_EPISODE):

                action = agent.select_action(
                    state,
                    noise_std=EXPLORATION_STD
                )

                (
                    next_state,
                    reward,
                    terminated,
                    truncated,
                    info
                ) = env.step(action)

                # Terminated states and time-limit
                # states both end this finite episode.
                done = terminated or truncated

                agent.replay.add(
                    state,
                    action,
                    reward,
                    next_state,
                    done
                )

                losses = agent.train_step()

                if losses is not None:
                    actor_losses.append(losses[0])
                    critic_losses.append(losses[1])

                state = next_state
                total_reward += reward

                if (
                    info["stable"]
                    and info["cost"] < best_cost
                ):
                    best_cost = info["cost"]
                    best_parameters = (
                        info["parameters"].copy()
                    )

                    np.save(
                        OUTPUT_DIR / "best_parameters.npy",
                        best_parameters
                    )

                if done:
                    break

            history.append({
                "episode": episode + 1,
                "final_cost": info["cost"],
                "total_reward": total_reward,
                "best_cost": best_cost,
                "knee_rmse": info["knee_rmse"],
                "ankle_rmse": info["ankle_rmse"],
                "actor_loss": (
                    np.mean(actor_losses)
                    if actor_losses else np.nan
                ),
                "critic_loss": (
                    np.mean(critic_losses)
                    if critic_losses else np.nan
                )
            })

            print(
                f"Episode {episode + 1:03d} | "
                f"Final cost: {info['cost']:.6f} | "
                f"Best: {best_cost:.6f} | "
                f"Reward: {total_reward:.4f}"
            )

        # Save final actor and critic.
        agent.actor.save(
            OUTPUT_DIR / "actor.keras"
        )

        agent.critic.save(
            OUTPUT_DIR / "critic.keras"
        )

        # Save best controller even if no improvement.
        np.save(
            OUTPUT_DIR / "best_parameters.npy",
            best_parameters
        )

        with open(
            OUTPUT_DIR / "training_history.csv",
            "w",
            newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(history[0].keys())
            )

            writer.writeheader()
            writer.writerows(history)

        improvement = (
            (BASELINE_COST - best_cost)
            / BASELINE_COST * 100.0
        )

        print("\n========== TRAINING COMPLETE ==========")
        print(f"Baseline cost: {BASELINE_COST:.6f}")
        print(f"Best cost: {best_cost:.6f}")
        print(f"Improvement: {improvement:.3f}%")
        print("Best controller:", best_parameters)
        print("Results saved to:", OUTPUT_DIR)

    finally:
        env.close()


if __name__ == "__main__":
    main()
