# %%
import gymnasium as gym
import numpy as np
import networkx as nx
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import torch

# Check if MPS is available
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
device = torch.device("mps" if mps_available else "cpu")
print(f"Using device: {device}")

class MISEnv(gym.Env):
    def __init__(self, num_nodes):
        super(MISEnv, self).__init__()
        self.num_nodes = num_nodes
        self.graph = None
        self.mis = set()
        self.action_space = gym.spaces.Discrete(num_nodes)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(num_nodes * 2,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.graph = nx.gnp_random_graph(self.num_nodes, 0.5)
        self.mis = set()
        return self._get_observation(), {}

    def step(self, action):
        node = action
        reward = 0
        
        if node not in self.mis and all(neighbor not in self.mis for neighbor in self.graph.neighbors(node)):
            self.mis.add(node)
            reward = 1
        
        done = len(self.mis) + len(set.union(*[set(self.graph.neighbors(n)) for n in self.mis])) == self.num_nodes
        
        return self._get_observation(), reward, done, False, {}

    def _get_observation(self):
        obs = np.zeros(self.num_nodes * 2, dtype=np.float32)
        obs[:self.num_nodes] = nx.to_numpy_array(self.graph).sum(axis=1) / (self.num_nodes - 1)  # Normalized degree
        obs[self.num_nodes:] = [1 if i in self.mis else 0 for i in range(self.num_nodes)]  # MIS membership
        return obs

def create_env(num_nodes):
    return lambda: MISEnv(num_nodes)

def train_agent(env, total_timesteps=100000):
    model = PPO("MlpPolicy", env, verbose=1, 
                learning_rate=0.0003, 
                n_steps=2048, 
                batch_size=64, 
                n_epochs=10, 
                gamma=0.99, 
                gae_lambda=0.95, 
                clip_range=0.2, 
                ent_coef=0.01,
                device=device)
    
    model.learn(total_timesteps=total_timesteps)
    return model

def evaluate_agent(model, env, num_episodes=100):
    mis_sizes = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
        mis_sizes.append(len(env.mis))
    return np.mean(mis_sizes), np.std(mis_sizes)

def visualize_mis(graph, mis):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=700)
    nx.draw_networkx_nodes(graph, pos, nodelist=list(mis), node_color='red', node_size=800)
    plt.title(f"Graph with MIS (size: {len(mis)})")
    plt.show()



# %%
NUM_NODES = 30
env = DummyVecEnv([create_env(NUM_NODES)])

print("Training agent...")
model = train_agent(env, total_timesteps=200000)

print("Evaluating agent...")
mean_mis_size, std_mis_size = evaluate_agent(model, env.envs[0])
print(f"Average MIS size: {mean_mis_size:.2f} ± {std_mis_size:.2f}")

print("Generating example MIS...")
obs, _ = env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)

final_graph = env.envs[0].graph
final_mis = env.envs[0].mis

print(f"Final MIS size: {len(final_mis)}")
print("Visualizing final graph with MIS...")
visualize_mis(final_graph, final_mis)

# Compare with NetworkX's implementation
nx_mis = nx.maximal_independent_set(final_graph)
print(f"NetworkX MIS size: {len(nx_mis)}")

# %%



