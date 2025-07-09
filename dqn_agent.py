import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from tqdm import tqdm
from blackjack_env import BlackjackEnv

# === DQN Network ===
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# === Hyperparameters ===
GAMMA = 0.99
BATCH_SIZE = 64
REPLAY_SIZE = 100_000
MIN_REPLAY_SIZE = 10_000
EPSILON_DECAY = 2_000_000
MIN_EPSILON = 0.05
LR = 1e-5
TARGET_UPDATE_FREQ = 10_000
TRAINING_STEPS = 500_000

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

env = BlackjackEnv()
state, _, _ = env.reset()
state_dim = len(state)
action_dim = 4

policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
replay_buffer = deque(maxlen=REPLAY_SIZE)

# === Epsilon decay ===
def epsilon_by_frame(frame):
    return max(MIN_EPSILON, 1.0 - frame / EPSILON_DECAY)

# === Pre-fill buffer ===
state, _, _ = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    action = random.choice(env.valid_actions())
    next_state, reward, done = env.step(action)
    replay_buffer.append((state, action, reward, next_state, done))
    state = next_state if not done else env.reset()[0]

def train():
    episode_rewards = []
    state, _, _ = env.reset()
    total_reward = 0.0

    for step in tqdm(range(TRAINING_STEPS), desc="Training"):
        epsilon = epsilon_by_frame(step)
        valid_actions = env.valid_actions()

        if random.random() < epsilon:
            action = random.choice(valid_actions)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
            valid_q = q_values[0][valid_actions]
            action = valid_actions[valid_q.argmax().item()]

        next_state, reward, done = env.step(action)
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if done:
            episode_rewards.append(total_reward)
            total_reward = 0.0
            state, _, _ = env.reset()

        batch = random.sample(replay_buffer, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.bool).to(device)

        q_values = policy_net(states).gather(1, actions).squeeze()
        with torch.no_grad():
            next_q_values = target_net(next_states).max(1)[0]
            targets = rewards + GAMMA * next_q_values * (~dones)

        loss = nn.MSELoss()(q_values, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if step % 5000 == 0 and step > 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Step {step}, Avg Reward (last 50): {avg_reward:.3f}, Epsilon: {epsilon:.4f}")

    torch.save(policy_net.state_dict(), "results/dqn_blackjack.pth")
    print("Model saved.")

if __name__ == "__main__":
    train()