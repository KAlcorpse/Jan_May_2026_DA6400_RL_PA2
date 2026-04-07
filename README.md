# Programming Assignment 2

## Deep Q-Networks on Mountain Car

This repository contains the solutions for **DA6400: Reinforcement Learning (Jan-May 2026) Programming Assignment 2**. 
The assignment focuses on **MountainCar-v0** and studies vanilla DQN, replay-factor variants, episode truncation, distributional performance analysis, tolerance intervals, sensitivity analysis, and a bonus experiment with Prioritized Experience Replay (PER). The environment must use **discrete actions**, **$\gamma = 0.99$**, **no reward shaping**, and a modified truncation length of **2000 timesteps** for the main experiments. 

------------------------------------------------------------------------

# Contributions

1. NA23B040 - Chandran K - Section - 3 (3, 4 (c), 5(Bonus))
2. ED23B051 - Kalyan S - Section - 3 (4(a), 4(b), 4(d))
3. EE23B004 - Amritam H - Section - 3 (1, 2)

------------------------------------------------------------------------

# Installation

To install the required libraries inside a runtime (for example **Google Colab**), run:

```python
!pip install -r requirements.txt
```

------------------------------------------------------------------------

# 1. Environment and Setup

We use the Gymnasium environment:

`MountainCar-v0`

Key settings used in the assignment:

- Discrete action space
- Continuous observation space
- Discount factor: `gamma = 0.99`
- No reward shaping
- Episode truncation lengths compared in the experiments: `200`, `1000`, and `2000`

The default DQN experiments use a truncation length of **2000 timesteps**. 

------------------------------------------------------------------------

# 2. Vanilla DQN on MountainCar-v0

This part implements a basic **Deep Q-Network (DQN)** with:

- $\epsilon$-greedy exploration
- fixed replay buffer
- hard target network updates
- uniform replay sampling
- Adam optimizer
- Kaiming initialization for the Q-network

The code is run over multiple random seeds and the mean performance is reported with confidence intervals.

------------------------------------------------------------------------

# 3. Effect of Episode Truncation

This section compares DQN performance under different episode truncation lengths:

- `200` timesteps
- `1000` timesteps
- `2000` timesteps

The goal is to study how truncation affects learning stability, return, and convergence behavior.

------------------------------------------------------------------------

# 4. Replay Factor Variants of DQN

Here, the replay factor $\rho$ is varied while keeping all other hyperparameters fixed.

The values used are:

- $\rho \in \{1, 2, 4, 8\}$

The experiments compare:

- mean learning curves
- aggregate performance
- distribution of performance
- tolerance intervals

------------------------------------------------------------------------

# 5. Distribution of Performance

This section visualizes the distribution of aggregate performance across runs for different replay factors. It helps compare:

- unimodality / multimodality
- skewness
- variance across seeds
- reliability of each setting

------------------------------------------------------------------------

# 6. Variability in Performance

This section plots mean performance with **tolerance intervals** to study the robustness of the replay-factor variants.

The analysis focuses on:

- reliability
- worst-case performance
- comparison with confidence intervals

------------------------------------------------------------------------

# 7. Sensitivity Analysis

This section studies the effect of two hyperparameters for DQN with $\rho \in \{1,4\}$:

- mini-batch size
- target network refresh rate

The batch sizes and target-update values are varied around the default setting to study robustness and sensitivity.

------------------------------------------------------------------------

# 8. Bonus: Prioritized Experience Replay

The bonus experiment replaces uniform replay with **Prioritized Experience Replay (PER)** and checks whether prioritizing samples changes the effect of increasing the replay factor.

------------------------------------------------------------------------

# Running the Experiments

Run the code sequentially:

1. Install dependencies
2. Train DQN for the required seeds
3. Save the logs / pickle files
4. Plot learning curves, distributions, and sensitivity plots
5. Report the results in the final document

------------------------------------------------------------------------

# Repository Structure

```text
.
├── requirements.txt
├── README.md
├── results/
└── notebooks / scripts for DQN and plots
```

------------------------------------------------------------------------
