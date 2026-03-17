import time

import torch
from torch import nn
from torch.nn import Conv2d, ReLU, AvgPool2d as Avg, Linear,Module,Sequential, AdaptiveAvgPool2d ,Flatten
import torchvision.transforms as transforms
import random
from collections import deque
import torch.optim as optim
import numpy as np

import environment


class Hw2Env(environment.BaseEnv):
    def __init__(self, n_actions=8, **kwargs) -> None:
        super().__init__(**kwargs)
        # divide the action space into n_actions
        self._n_actions = n_actions
        self._delta = 0.05

        theta = np.linspace(0, 2*np.pi, n_actions)
        actions = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        self._actions = {i: action for i, action in enumerate(actions)}

        self._goal_thresh = 0.01
        self._max_timesteps = 50

    def _create_scene(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        scene = environment.create_tabletop_scene()
        obj_pos = [np.random.uniform(0.25, 0.75),
                   np.random.uniform(-0.3, 0.3),
                   1.5]
        goal_pos = [np.random.uniform(0.25, 0.75),
                    np.random.uniform(-0.3, 0.3),
                    1.025]
        environment.create_object(scene, "box", pos=obj_pos, quat=[0, 0, 0, 1],
                                  size=[0.03, 0.03, 0.03], rgba=[0.8, 0.2, 0.2, 1],
                                  name="obj1")
        environment.create_visual(scene, "cylinder", pos=goal_pos, quat=[0, 0, 0, 1],
                                  size=[0.05, 0.005], rgba=[0.2, 1.0, 0.2, 1],
                                  name="goal")
        return scene

    def state(self):
        if self._render_mode == "offscreen":
            self.viewer.update_scene(self.data, camera="topdown")
            pixels = torch.tensor(self.viewer.render().copy(), dtype=torch.uint8).permute(2, 0, 1)
        else:
            pixels = self.viewer.read_pixels(camid=1).copy()
            pixels = torch.tensor(pixels, dtype=torch.uint8).permute(2, 0, 1)
            pixels = transforms.functional.center_crop(pixels, min(pixels.shape[1:]))
            pixels = transforms.functional.resize(pixels, (128, 128))
        return pixels / 255.0

    def high_level_state(self):
        ee_pos = self.data.site(self._ee_site).xpos[:2]
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.concatenate([ee_pos, obj_pos, goal_pos])

    def reward(self):
        state = self.high_level_state()
        ee_pos = state[:2]
        obj_pos = state[2:4]
        goal_pos = state[4:6]
        ee_to_obj = max(100*np.linalg.norm(ee_pos - obj_pos), 1)
        obj_to_goal = max(100*np.linalg.norm(obj_pos - goal_pos), 1)
        return 1/(ee_to_obj) + 1/(obj_to_goal)

    def is_terminal(self):
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.linalg.norm(obj_pos - goal_pos) < self._goal_thresh

    def is_truncated(self):
        return self._t >= self._max_timesteps

    def step(self, action_id):
        action = self._actions[action_id] * self._delta
        ee_pos = self.data.site(self._ee_site).xpos[:2]
        target_pos = np.concatenate([ee_pos, [1.06]])
        target_pos[:2] = np.clip(target_pos[:2] + action, [0.25, -0.3], [0.75, 0.3])
        self._set_ee_in_cartesian(target_pos, rotation=[-90, 0, 180], n_splits=30, threshold=0.04)
        self._t += 1

        state = self.state()
        reward = self.reward()
        terminal = self.is_terminal()
        truncated = self.is_truncated()
        return state, reward, terminal, truncated


class Network(Module): # just modified the code from description to make an inirialization function and a forward function
    def __init__(self, n_actions):
        super().__init__()

        self.model = Sequential(
            Conv2d(3, 32, 4, 2, 1), ReLU(),
            Conv2d(32, 64, 4, 2, 1), ReLU(),
            Conv2d(64, 128, 4, 2, 1), ReLU(),
            Conv2d(128, 256, 4, 2, 1), ReLU(),
            Conv2d(256, 512, 4, 2, 1), ReLU(),
            AdaptiveAvgPool2d((1, 1)),
            Flatten(),  
            Linear(512, n_actions)
        )

    def forward(self, x):
        return self.model(x)

N_ACTIONS = 8

net = Network(N_ACTIONS)

GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.999 # decay epsilon by 0.999 every EPSILON_DECAY_ITER
EPSILON_DECAY_ITER = 10 # decay epsilon every 100 updates
MIN_EPSILON = 0.1 # minimum epsilon
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
UPDATE_FREQ = 4 # update the network every 4 steps
TARGET_NETWORK_UPDATE_FREQ = 100 # update the target network every 1000 steps
BUFFER_LENGTH = 10000
NUMBER_OF_EPISODES = 500

replay_buffer = deque(maxlen=BUFFER_LENGTH)

def push_transition(state, action, reward, next_state, done):
    replay_buffer.append((state, action, reward, next_state, done))

def sample_batch(batch_size):
    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    return (
        torch.stack(states).float(),                # shape: (B, 3, 128,128)
        torch.tensor(actions, dtype=torch.long),    # shape: (B,)
        torch.tensor(rewards).float(),              # shape: (B,)
        torch.stack(next_states).float(),           # shape: (B, 3, 128,128)
        torch.tensor(dones).float()                 # shape: (B,)
    )
    
    
target_net = Network(N_ACTIONS)
target_net.load_state_dict(net.state_dict())
target_net.eval()

optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()
        
        
def train_step():
    if len(replay_buffer) < BATCH_SIZE:
        return  # not enough samples

    states, actions, rewards, next_states, dones = sample_batch(BATCH_SIZE)

    # Compute Q-values for current states
    q_values = net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute target Q-values
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0]
        target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

    # Loss and backprop
    loss = loss_fn(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
        
if __name__ == "__main__":
    N_ACTIONS = 8
    env = Hw2Env(n_actions=N_ACTIONS, render_mode="offscreen") # to not render the environment during training
    for episode in range(NUMBER_OF_EPISODES):
        env.reset()
        state = env.state()
        done = False
        cumulative_reward = 0.0
        episode_steps = 0
        start = time.time()
        while not done:
            # ε-greedy action selection
            if random.random() < EPSILON:
                action = np.random.randint(N_ACTIONS)
            else:
                with torch.no_grad():
                    q_values = net(state.unsqueeze(0).float())
                    action = q_values.argmax().item()

            next_state, reward, is_terminal, is_truncated = env.step(action)
            done = is_terminal or is_truncated

            # store transition
            push_transition(state.detach(), action, reward, next_state.detach(), done) # detach the tensors to avoid backprop through the environment

            state = next_state
            cumulative_reward += reward
            episode_steps += 1

            # train periodically
            if episode_steps % UPDATE_FREQ == 0:
                train_step()

            # update target network
            if episode_steps % TARGET_NETWORK_UPDATE_FREQ == 0:
                target_net.load_state_dict(net.state_dict())
                
                
        end = time.time()
        print(f"Episode={episode}, reward={cumulative_reward}, RPS={cumulative_reward/episode_steps}")
        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
# for visualising the final model

    print("Training finished. Running visual test...")

    env = Hw2Env(n_actions=N_ACTIONS, render_mode="gui")
    env.reset()

    state = env.state()
    done = False

    while not done:
        with torch.no_grad():
            q_values = net(state.unsqueeze(0).float())
            action = q_values.argmax().item()

        state, reward, is_terminal, is_truncated = env.step(action)
        done = is_terminal or is_truncated