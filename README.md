# 🃏 Blackjack DQN Agent

This project implements a Deep Q-Network (DQN) reinforcement learning agent to play Blackjack. The agent learns optimal actions like **hit**, **stand**, **double**, and **split** using a custom environment with realistic card counting over 6 decks.

---

## 🚀 Features

- Custom Blackjack environment with:
  - Dealer rules (hits on soft 17)
  - 6-deck simulation with card tracking
  - Natural blackjack detection
  - Support for double and split
- DQN agent using PyTorch
- Evaluation over 100,000 simulated games
- GPU-accelerated training
- Environment exported to `environment.yml` for easy reproducibility

---

## 🛠 Installation

Clone the repository:

```bash
git clone https://github.com/MH261102/blackjack-dqn.git
cd blackjack-dqn
```

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate blackjack-dqn
```

---

## 📂 Project Structure

```text
.
├── blackjack_env.py       # Custom Blackjack environment
├── dqn_agent.py           # DQN agent training script
├── evaluate_dqn.py        # Evaluation script for trained agent
├── environment.yml        # Conda environment definition
├── results/               # Folder where the trained model is saved
└── __pycache__/           # Python cache files (ignored)
```

---

## 🧠 Usage

### Training

```bash
python dqn_agent.py
```

The model will be saved to `results/dqn_blackjack.pth` after training.

### Evaluation

```bash
python evaluate_dqn.py
```

Outputs win/loss statistics and action usage over 100,000 games.

---

## 📈 Sample Evaluation Output

```
Evaluation over 100000 games:
Win Rate:  38.47%
Tie Rate:  8.13%
Loss Rate: 53.40%

Action Usage:
Hit:    86.57%
Stand:  69.00%
Double:  1.42%
Split:  11.53%
```

---

## 📋 To Do / Future Improvements

- Support multiple simultaneous hands after split
- Add counting strategy analysis (Hi-Lo)
- Experiment with Double DQN or Dueling DQN
- Add visualizations of Q-values or training progression

---

## 📜 License

MIT License – free to use, modify, and share.

---

## 🤝 Acknowledgements

Built with PyTorch.
