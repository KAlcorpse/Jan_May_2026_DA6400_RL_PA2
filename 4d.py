import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
import pickle
from scipy import stats
import os


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim)
        )

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
    


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, done = zip(*batch)
        return (
            torch.tensor(np.array(s), dtype=torch.float32),
            torch.tensor(np.array(a), dtype=torch.long),
            torch.tensor(np.array(r), dtype=torch.float32),
            torch.tensor(np.array(s_next), dtype=torch.float32),
            torch.tensor(np.array(done), dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)
    


REPLAY_FACTOR   = [1, 2, 4, 8]
TRUNCATION_LEN  = 2000
BUFFER_CAPACITY = 50000
BATCH_SIZE      = 64
GAMMA           = 0.99
LR              = 1e-3
EPSILON_START   = 1.0
EPSILON_END     = 0.05
EPSILON_DECAY   = 50000
TARGET_UPDATE   = 500
MAX_TIMESTEPS   = 200000
MIN_BUFFER_SIZE = 500
PRINT_EVERY     = 10
TARGET_REWARD   = -100


BATCH_SIZES     = [16, 32, 64, 128, 256]
TARGET_UPDATES  = [100, 250, 500, 1000, 2000]
RHO             = [1, 4]
SEEDS           = list(range(10))



# Train function
def train_sensitivity(seed, rep_fact, batch_size, target_update):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env    = gym.make("MountainCar-v0", max_episode_steps=TRUNCATION_LEN)

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net      = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    buffer    = ReplayBuffer(BUFFER_CAPACITY)

    epsilon        = EPSILON_START
    step_count     = 0
    returns        = []
    episode_return = 0

    state, _ = env.reset(seed=seed)

    while step_count < MAX_TIMESTEPS:

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                st = torch.from_numpy(state).float().unsqueeze(0).to(device)
                action = q_net(st).argmax().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.push(state, action, reward, next_state, float(terminated))
        state           = next_state
        episode_return += reward
        step_count     += 1

        epsilon = max(EPSILON_END, EPSILON_START - step_count / EPSILON_DECAY)

        if len(buffer) > MIN_BUFFER_SIZE:
            for _ in range(rep_fact):
                s, a, r, s_next, d = buffer.sample(batch_size)

                s      = torch.from_numpy(np.array(s)).float().to(device)
                a      = torch.from_numpy(np.array(a)).long().to(device)
                r      = torch.from_numpy(np.array(r)).float().to(device)
                s_next = torch.from_numpy(np.array(s_next)).float().to(device)
                d      = torch.from_numpy(np.array(d)).float().to(device)

                q_values = q_net(s).gather(1, a.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    target = r + GAMMA * target_net(s_next).max(1)[0] * (1 - d)

                loss = nn.MSELoss()(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if step_count % target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

        if done:
            returns.append(episode_return)
            if len(returns) % PRINT_EVERY == 0:
                print(
                    f"[ ρ = {rep_fact} | seed = {seed} ] "
                    f"Episode: {len(returns):>5} | "
                    f"ε: {epsilon:.3f} | "
                    f"Best: {max(returns):>8.1f} | "
                    f"Mean (50): {np.mean(returns[-50:]):>8.1f}"
                )

            state, _       = env.reset()
            episode_return = 0



    env.close()

    results  = {
        "returns":       returns,
        "replay_factor": rep_fact,
        "batch_size":    batch_size,
        "target_update": target_update,
        "seed":          seed,
    }


    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)



    filename = f"sens_rho{rep_fact}_bs{batch_size}_tu{target_update}_seed{seed}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(results, f)




    return returns


print("-- Target Update Sensitivity --")
print("Device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
for rho in RHO:
    for tu in TARGET_UPDATES:
        for seed in SEEDS:
            print(f"ρ={rho} | target_update={tu} | seed={seed}")
            train_sensitivity(seed, rho, BATCH_SIZE, tu)