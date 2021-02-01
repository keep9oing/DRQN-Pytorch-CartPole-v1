import sys
from typing import Dict, List, Tuple

import gym
import collections
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

# from SegmentTree import MinSegmentTree, SumSegmentTree


# Q_network
class Q_net(nn.Module):
    def __init__(self, state_space=None,
                 action_space=None):
        super(Q_net, self).__init__()

        # space size check
        assert state_space is not None, "None state_space input: state_space should be selected."
        assert action_space is not None, "None action_space input: action_space should be selected."

        self.Linear1 = nn.Linear(state_space, 64)
        # self.lstm    = nn.LSTMCell(64,64)
        self.Linear2 = nn.Linear(64, 64)
        self.Linear3 = nn.Linear(64, action_space)

    def forward(self, x):
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        return self.Linear3(x)

    def sample_action(self, obs, epsilon):
        if random.random() < epsilon:
            return random.randint(0,1)
        else:
            return self.forward(obs).argmax().item()

class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def put(
        self,
        obs: np.ndarray,
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size


def train(q_net=None, target_q_net=None, replay_buffer=None,
          device=None, 
          optimizer = None,
          batch_size=64,
          learning_rate=1e-3,
          gamma=0.99):

    assert device is not None, "None Device input: device should be selected."

    # Get batch from replay buffer
    samples = replay_buffer.sample()

    
    states = torch.FloatTensor(samples["obs"]).to(device)
    actions = torch.LongTensor(samples["acts"].reshape(-1,1)).to(device)
    rewards = torch.FloatTensor(samples["rews"].reshape(-1,1)).to(device)
    next_states = torch.FloatTensor(samples["next_obs"]).to(device)
    dones = torch.FloatTensor(samples["done"].reshape(-1,1)).to(device)

    # Define loss
    q_target_max = target_q_net(next_states).max(1)[0].unsqueeze(1).detach()
    targets = rewards + gamma*q_target_max*dones
    q_out = q_net(states)
    q_a = q_out.gather(1, actions)

    # Multiply Importance Sampling weights to loss        
    loss = F.smooth_l1_loss(q_a, targets)
    
    # Update Network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def seed_torch(seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

def save_model(model, path='default.pth'):
        torch.save(model.state_dict(), path)

if __name__ == "__main__":

    # Determine seeds
    model_name = "DQN_POMDP"
    env_name = "CartPole-v1"
    seed = 1
    exp_num = 'SEED'+'_'+str(seed)

    # Set gym environment
    env = gym.make(env_name)

    if torch.cuda.is_available():
        device = torch.device("cuda")

    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)
    env.seed(seed)

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('runs/'+env_name+"_"+model_name+"_"+exp_num)

    # Set parameters
    batch_size = 64
    learning_rate = 1e-3
    buffer_len = int(100000)
    min_buffer_len = batch_size
    episodes = 650
    print_per_iter = 20
    target_update_period = 4
    eps_start = 0.1
    eps_end = 0.001
    eps_decay = 0.995
    tau = 1*1e-2
    max_step = 2000

    # Create Q functions
    Q = Q_net(state_space=env.observation_space.shape[0]-2, 
              action_space=env.action_space.n).to(device)
    Q_target = Q_net(state_space=env.observation_space.shape[0]-2, 
                     action_space=env.action_space.n).to(device)

    Q_target.load_state_dict(Q.state_dict())

    # Create Replay buffer
    replay_buffer = ReplayBuffer(env.observation_space.shape[0]-2,
                                            size=buffer_len, batch_size=batch_size)

    # Set optimizer
    score = 0
    score_sum = 0
    optimizer = optim.Adam(Q.parameters(), lr=learning_rate)

    epsilon = eps_start

    # Train
    for i in range(episodes):
        s = env.reset()
        s = s[::2] # Use only Position of Cart and Pole
        done = False
        
        for t in range(max_step):
            # if i % print_per_iter == 0:
            #     env.render()

            # Get action
            a = Q.sample_action(torch.from_numpy(s).float().to(device), epsilon)

            # Do action
            s_prime, r, done, _ = env.step(a)
            s_prime = s_prime[::2]
            # r += s_prime[0] ## For MountainCar

            # make data
            done_mask = 0.0 if done else 1.0

            replay_buffer.put(s, a, r/100.0, s_prime, done_mask)

            s = s_prime
            
            score += r
            score_sum += r

            if len(replay_buffer) >= min_buffer_len:
                train(Q, Q_target, replay_buffer, device, 
                        optimizer=optimizer,
                        batch_size=batch_size,
                        learning_rate=learning_rate)

                if (t+1) % target_update_period == 0:
                    # Q_target.load_state_dict(Q.state_dict())
                    for target_param, local_param in zip(Q_target.parameters(), Q.parameters()):
                            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)
                
            if done:
                break
        
        epsilon = max(eps_end, epsilon * eps_decay) #Linear annealing

        if i % print_per_iter == 0 and i!=0:
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            i, score_sum/print_per_iter, len(replay_buffer), epsilon*100))
            score_sum=0.0
            save_model(Q, model_name+"_"+exp_num+'.pth')

        # Log the reward
        writer.add_scalar('Rewards per episodes', score, i)
        score = 0
        
    writer.close()
    env.close()