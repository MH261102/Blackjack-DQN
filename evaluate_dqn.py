import torch
import numpy as np
from blackjack_env import BlackjackEnv
from dqn_agent import DQN  # make sure DQN class is defined in dqn_agent.py
from tqdm import tqdm

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === Environment and Model ===
env = BlackjackEnv()
state, _, _ = env.reset()  # returns state, reward, done
state_dim = len(state)
action_dim = 4

model = DQN(state_dim, action_dim).to(device)
model.load_state_dict(torch.load("results/dqn_blackjack.pth"))
model.eval()

# === Evaluation Metrics ===
win, tie, loss = 0, 0, 0
action_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # hit, stand, double, split

# === Evaluation Loop ===
for _ in tqdm(range(100_000), desc="Evaluating"):
    state, _, done = env.reset()
    
    while not done:
        valid_actions = env.valid_actions()
        state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            q_values = model(state_tensor).squeeze().cpu().numpy()

        action = max(valid_actions, key=lambda a: q_values[a])
        action_counts[action] += 1

        state, reward, done = env.step(action)

    if reward > 0:
        win += 1
    elif reward == 0:
        tie += 1
    else:
        loss += 1

# === Print Results ===
total = win + tie + loss
print("\nEvaluation over 100000 games:")
print("Win Rate: ", f"{win/total*100:.2f}%")
print("Tie Rate: ", f"{tie/total*100:.2f}%")
print("Loss Rate:", f"{loss/total*100:.2f}%")

print("\nAction Usage:")
actions_map = {0: "Hit", 1: "Stand", 2: "Double", 3: "Split"}
for action, count in action_counts.items():
    print(f"{actions_map[action]}: {count} times ({count/total*100:.2f}%)")
