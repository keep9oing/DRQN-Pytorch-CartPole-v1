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


# Q_network
class Q_net(nn.Module):
    def __init__(self, state_space=None,
                 action_space=None):
        super(Q_net, self).__init__()

        # space size check
        assert state_space is not None, "None state_space input: state_space should be selected."
        assert action_space is not None, "None action_space input: action_space should be selected."

        self.hidden_space = 64
        self.state_space = state_space
        self.action_space = action_space

        self.Linear1 = nn.Linear(self.state_space, self.hidden_space)
        self.lstm    = nn.LSTM(self.hidden_space,self.hidden_space)
        self.Linear2 = nn.Linear(self.hidden_space, self.action_space)

    def forward(self, x, h, c):
        x = F.relu(self.Linear1(x))
        x, (new_h, new_c) = self.lstm(x,(h,c))
        x = self.Linear2(x)
        return x.squeeze(), new_h, new_c

    def sample_action(self, obs, h,c, epsilon):
        output = self.forward(obs, h,c)

        if random.random() < epsilon:
            return random.randint(0,1), output[1], output[2]
        else:
            return output[0].argmax().item(), output[1] , output[2]
    
    def init_hidden_state(self, batch_size):
        return torch.zeros([1,batch_size, self.hidden_space]), torch.zeros([1, batch_size, self.hidden_space])

class EpisodeMemory():
    """Episode memory for recurrent agent"""

    def __init__(self, random_update=False, 
                       max_epi_num=100, max_epi_len=500,
                       batch_size=1):
        self.random_update = random_update # if False, sequential update
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.batch_size = batch_size

        if (random_update is False) and (self.batch_size > 1):
            sys.exit('It is recommend to use 1 batch for sequential update, if you want, erase this code block and modify code')

        self.memory = collections.deque(maxlen=self.max_epi_num)

    def put(self, episode):
        self.memory.append(episode)

    def sample(self):
        sampled_episodes = []

        if self.random_update: # Random upodate
            pass
        else: # Sequential update
            idx = np.random.randint(0, len(self.memory))
            sampled_episodes.append(self.memory[idx].sample())

        return sampled_episodes

    def __len__(self):
        return len(self.memory)


class EpisodeBuffer:
    """A simple numpy replay buffer."""

    def __init__(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.next_obs = []
        self.done = []

    def put(self, transition):
        self.obs.append(transition[0])
        self.action.append(transition[1])
        self.reward.append(transition[2])
        self.next_obs.append(transition[3])
        self.done.append(transition[4])

    def sample(self) -> Dict[str, np.ndarray]:
        obs = np.array(self.obs)
        action = np.array(self.action)
        reward = np.array(self.reward)
        next_obs = np.array(self.next_obs)
        done = np.array(self.done)

        return dict(obs=obs,
                    acts=action,
                    rews=reward,
                    next_obs=next_obs,
                    done=done)

    def __len__(self) -> int:
        return len(self.obs)


def train(q_net=None, target_q_net=None, episode_memory=None,
          device=None, 
          optimizer = None,
          batch_size=1,
          learning_rate=1e-3,
          gamma=0.99):

    assert device is not None, "None Device input: device should be selected."

    # Get batch from replay buffer
    samples = episode_memory.sample()

    for i in range(batch_size):
        observations = torch.FloatTensor(samples[i]["obs"]).to(device)
        actions = torch.LongTensor(samples[i]["acts"].reshape(-1,1)).to(device)
        rewards = torch.FloatTensor(samples[i]["rews"].reshape(-1,1)).to(device)
        next_observations = torch.FloatTensor(samples[i]["next_obs"]).to(device)
        dones = torch.FloatTensor(samples[i]["done"].reshape(-1,1)).to(device)

    h_target, c_target = target_q_net.init_hidden_state(batch_size)
    # Define loss
    if episode_memory.random_update:
        pass
    else:
        q_target, _, _ = target_q_net(next_observations.unsqueeze(1), h_target.to(device), c_target.to(device))

    q_target_max = q_target.max(1)[0].unsqueeze(1).detach()
    targets = rewards + gamma*q_target_max*dones


    h, c = q_net.init_hidden_state(batch_size)
    q_out, _, _ = q_net(observations.unsqueeze(1), h.to(device), c.to(device))
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
    model_name = "DRQN_POMDP_SEQ"
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
    batch_size = 1
    learning_rate = 1e-3
    buffer_len = int(100000)
    min_epi_num = 10
    episodes = 650
    print_per_iter = 20
    target_update_period = 4
    eps_start = 0.1
    eps_end = 0.001
    eps_decay = 0.995
    tau = 1e-2
    max_step = 2000

    # DRQN param
    update_step = 5 # If you want to do random update instead of sequential update

    # Create Q functions
    Q = Q_net(state_space=env.observation_space.shape[0]-2, 
              action_space=env.action_space.n).to(device)
    Q_target = Q_net(state_space=env.observation_space.shape[0]-2, 
                     action_space=env.action_space.n).to(device)

    Q_target.load_state_dict(Q.state_dict())

    # Set optimizer
    score = 0
    score_sum = 0
    optimizer = optim.Adam(Q.parameters(), lr=learning_rate)

    epsilon = eps_start
    
    episode_memory = EpisodeMemory(random_update=False, max_epi_num=100, max_epi_len=600, batch_size=batch_size)

    # Train
    for i in range(episodes):
        s = env.reset()
        obs = s[::2] # Use only Position of Cart and Pole
        done = False
        
        episode_record = EpisodeBuffer()
        h, c = Q.init_hidden_state(batch_size)

        for t in range(max_step):

            # Get action
            a, h, c = Q.sample_action(torch.from_numpy(obs).float().to(device).unsqueeze(0).unsqueeze(0), 
                                              h.to(device), c.to(device),
                                              epsilon)

            # Do action
            s_prime, r, done, _ = env.step(a)
            obs_prime = s_prime[::2]

            # make data
            done_mask = 0.0 if done else 1.0

            episode_record.put([obs, a, r/100.0, obs_prime, done_mask])

            obs = obs_prime
            
            score += r
            score_sum += r

            if len(episode_memory) >= min_epi_num:
                train(Q, Q_target, episode_memory, device, 
                        optimizer=optimizer,
                        batch_size=batch_size,
                        learning_rate=learning_rate)

                if (t+1) % target_update_period == 0:
                    # Q_target.load_state_dict(Q.state_dict()) <- navie update
                    for target_param, local_param in zip(Q_target.parameters(), Q.parameters()): # <- soft update
                            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)
                
            if done:
                break
        
        episode_memory.put(episode_record)
        
        epsilon = max(eps_end, epsilon * eps_decay) #Linear annealing

        if i % print_per_iter == 0 and i!=0:
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            i, score_sum/print_per_iter, len(episode_memory), epsilon*100))
            score_sum=0.0
            save_model(Q, model_name+"_"+exp_num+'.pth')

        # Log the reward
        writer.add_scalar('Rewards per episodes', score, i)
        score = 0
        
    writer.close()
    env.close()