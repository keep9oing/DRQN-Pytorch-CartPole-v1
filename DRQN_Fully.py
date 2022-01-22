import sys
from typing import Dict, List, Tuple
from types import SimpleNamespace

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
    """
    Returns q_vals, h_ts, final_h on __call__
    """

    def __init__(self, args, state_space=None, action_space=None):
        super(Q_net, self).__init__()

        # space size check
        assert (
            state_space is not None
        ), "None state_space input: state_space should be selected."
        assert (
            action_space is not None
        ), "None action_space input: action_space should be selected."

        self.args = args
        self.hidden_space = 64
        self.state_space = state_space
        self.action_space = action_space

        self.Linear1 = nn.Linear(self.state_space, self.hidden_space)
        # self.lstm    = nn.LSTM(self.hidden_space,self.hidden_space, batch_first=True)
        self.GRU = nn.GRU(self.hidden_space, self.hidden_space, batch_first=True)
        self.Linear2 = nn.Linear(
            self.hidden_space, self.action_space
        )  # serves as QHead
        self.to(args.device)

    def forward(self, x, h):
        x = F.relu(self.Linear1(x))
        # x, (new_h, new_c) = self.lstm(x,(h,c)) # c was passed as param into function
        h_ts, final_h = self.GRU(x, h)
        q_vals = self.Linear2(h_ts)
        # return x, new_h,
        return q_vals, h_ts, final_h

    def sample_action(self, obs, h, epsilon):
        output = self.forward(obs, h)

        # if random.random() < epsilon:
        #     return random.randint(0,1), output[1], output[2]
        # else:
        #     return output[0].argmax().item(), output[1] , output[2]
        if random.random() < epsilon:
            return random.randint(0, 1), output[1]
        else:
            return output[0].argmax().item(), output[1]

    def init_hidden_state(self, batch_size, training=None):

        assert training is not None, "training step parameter should be dtermined"

        # if training is True:
        #     return torch.zeros([1, batch_size, self.hidden_space]), torch.zeros([1, batch_size, self.hidden_space])
        # else:
        #     return torch.zeros([1, 1, self.hidden_space]), torch.zeros([1, 1, self.hidden_space])
        if training is True:
            return torch.zeros([1, batch_size, self.hidden_space])
        else:
            return torch.zeros([1, 1, self.hidden_space])


class TransitionModel(nn.Module):
    def __init__(self, action_space, args):
        super().__init__()
        self.args = args
        layers = [
            nn.Linear(args.hidden_space + action_space, args.tran1_dim),
            nn.ReLU(),
            nn.Linear(args.tran1_dim, args.tran2_dim),
            nn.ReLU(),
            nn.Linear(args.tran2_dim, args.hidden_space),
        ]
        self.network = nn.Sequential(*layers)
        self.train()
        self.to(args.device)

    def forward(self, x):
        # stacked = torch.cat(x, action_onehot, dim=1)
        # stacking done externally
        next_state = self.network(x)

        return next_state


class ProjectionHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.fc1 = nn.Linear(args.hidden_space, 256)
        self.fc2 = nn.Linear(256, args.projection_out_dim)
        self.to(args.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class PredictionHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.fc1 = nn.Linear(args.projection_out_dim, args.prediction_dim)
        self.fc2 = nn.Linear(args.prediction_dim, args.projection_out_dim)
        self.to(args.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def soft_copy(target, source, tau):
    """
    Function performs exponential moving average copy.

    args:
    -----
        target (nn.Module): target network that will be a copy of source.
        source (nn.Module): the source network that serves as the copy.
        tau (float): controls the rate of copy.
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def freeze_params(model):
    """
    Freeze model params, dont want target encoder to accumulate gradients
    """
    for param in model.parameters():
        param.requires_grad = False


def hard_copy(target, source, freeze=True):
    """
    Do exact copy, freeze if boolean is passed
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

    # if freeze is True, we freeze gradients on the target model after copy
    if freeze:
        freeze_params(target)


def cosine_loss(x1, x2):
    criterion = torch.nn.CosineSimilarity(dim=1)
    loss = -criterion(x1, x2).mean()

    return loss


def L1_loss(x1, x2):
    loss = F.smooth_l1_loss(x1, x2)

    return loss


class EpisodeMemory:
    """Episode memory for recurrent agent"""

    def __init__(
        self,
        random_update=False,
        max_epi_num=100,
        max_epi_len=500,
        batch_size=1,
        lookup_step=None,
    ):
        self.random_update = random_update  # if False, sequential update
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.batch_size = batch_size
        self.lookup_step = lookup_step

        if (random_update is False) and (self.batch_size > 1):
            sys.exit(
                "It is recommend to use 1 batch for sequential update, if you want, erase this code block and modify code"
            )

        self.memory = collections.deque(maxlen=self.max_epi_num)

    def put(self, episode):
        self.memory.append(episode)

    def sample(self):
        sampled_buffer = []

        ##################### RANDOM UPDATE ############################
        if self.random_update:  # Random upodate
            sampled_episodes = random.sample(self.memory, self.batch_size)

            check_flag = (
                True  # check if every sample data to train is larger than batch size
            )
            min_step = self.max_epi_len

            for episode in sampled_episodes:
                min_step = min(
                    min_step, len(episode)
                )  # get minimum step from sampled episodes

            for episode in sampled_episodes:
                if min_step > self.lookup_step:  # sample buffer with lookup_step size
                    idx = np.random.randint(0, len(episode) - self.lookup_step + 1)
                    sample = episode.sample(
                        random_update=self.random_update,
                        lookup_step=self.lookup_step,
                        idx=idx,
                    )
                    sampled_buffer.append(sample)
                else:
                    idx = np.random.randint(
                        0, len(episode) - min_step + 1
                    )  # sample buffer with minstep size
                    sample = episode.sample(
                        random_update=self.random_update, lookup_step=min_step, idx=idx
                    )
                    sampled_buffer.append(sample)

        ##################### SEQUENTIAL UPDATE ############################
        else:  # Sequential update
            idx = np.random.randint(0, len(self.memory))
            sampled_buffer.append(
                self.memory[idx].sample(random_update=self.random_update)
            )

        return sampled_buffer, len(sampled_buffer[0]["obs"])  # buffers, sequence_length

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

    def sample(
        self, random_update=False, lookup_step=None, idx=None
    ) -> Dict[str, np.ndarray]:
        obs = np.array(self.obs)
        action = np.array(self.action)
        reward = np.array(self.reward)
        next_obs = np.array(self.next_obs)
        done = np.array(self.done)

        if random_update is True:
            obs = obs[idx : idx + lookup_step]
            action = action[idx : idx + lookup_step]
            reward = reward[idx : idx + lookup_step]
            next_obs = next_obs[idx : idx + lookup_step]
            done = done[idx : idx + lookup_step]

        return dict(obs=obs, acts=action, rews=reward, next_obs=next_obs, done=done)

    def __len__(self) -> int:
        return len(self.obs)


def spr_train(
    # q_net,
    # soft_q_net,
    projection,
    soft_projection,
    predictor,
    transition_model,
    hiddens,
    targets,
    actions,
    loss_function,
    args,
    k_steps=2,
):
    # NOTE: I'd rather pass in the targets, do it from outside? X: not feasible, need to time skip
    # soft_copy(soft_q_net, q_net, args.tau)
    # freeze_params(soft_q_net)
    # soft_copy(soft_projection, projection)
    # freeze_params(soft_projection)

    spr_loss = torch.tensor([0.0]).to(args.device)
    # DEBUG: hiddens are [64, 8, 64] and actions are [64, 8, 1] offending index 0 but its the last one that mismatches...
    stacked = torch.cat([hiddens, actions], 2)  # concat along time

    for t in range(1, k_steps + 1):
        next_encoding = transition_model(stacked)
        next_encoding = next_encoding[:, : -(t - 1)]  # discard from the back
        online_projection = projection(next_encoding)
        online_prediction = predictor(online_projection)

        # targets can be done outside since the encoder model does not change
        target_encoding = targets[
            :, t - 1 :
        ]  # t-1 slicing since targets generated from next_observations
        target_projection = soft_projection(target_encoding)

        # currently online_prediction is [64, 7, 16] and target_projection is [64, 8, 16]
        spr_loss_ = loss_function(online_prediction, target_projection)
        spr_loss += spr_loss_

    return spr_loss


def train(
    q_net=None,
    target_q_net=None,
    projection=None,
    soft_projection=None,
    predictor=None,
    transition_model=None,
    args=None,
    episode_memory=None,
    device=None,
    optimizer=None,
    batch_size=1,
    gamma=0.99,
):

    assert device is not None, "None Device input: device should be selected."

    # NOTE: No need separate soft copy Q-net because current RL implementation
    #      already makes use of soft copying at the same frequency we require
    soft_copy(soft_projection, projection, args.tau)

    # Get batch from replay buffer
    samples, seq_len = episode_memory.sample()

    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []

    for i in range(batch_size):
        observations.append(samples[i]["obs"])
        actions.append(samples[i]["acts"])
        rewards.append(samples[i]["rews"])
        next_observations.append(samples[i]["next_obs"])
        dones.append(samples[i]["done"])

    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_observations = np.array(next_observations)
    dones = np.array(dones)

    observations = torch.FloatTensor(observations.reshape(batch_size, seq_len, -1)).to(
        device
    )
    actions = torch.LongTensor(actions.reshape(batch_size, seq_len, -1)).to(device)
    rewards = torch.FloatTensor(rewards.reshape(batch_size, seq_len, -1)).to(device)
    next_observations = torch.FloatTensor(
        next_observations.reshape(batch_size, seq_len, -1)
    ).to(device)
    dones = torch.FloatTensor(dones.reshape(batch_size, seq_len, -1)).to(device)

    # h_target, c_target = target_q_net.init_hidden_state(batch_size=batch_size, training=True)
    h_target = target_q_net.init_hidden_state(batch_size=batch_size, training=True)

    q_target, h_target, _ = target_q_net(next_observations, h_target.to(device))

    q_target_max = q_target.max(2)[0].view(batch_size, seq_len, -1).detach()
    targets = rewards + gamma * q_target_max * dones

    h = q_net.init_hidden_state(batch_size=batch_size, training=True)
    q_out, h_ts, final_h = q_net(observations, h.to(device))
    q_a = q_out.gather(2, actions)

    # Multiply Importance Sampling weights to loss
    loss = F.smooth_l1_loss(q_a, targets)

    spr_loss = spr_train(
        projection,
        soft_projection,
        predictor,
        transition_model,
        h_ts,
        h_target,
        actions,
        cosine_loss,
        args,
        args.k_steps,
    )

    loss = loss + args.spr_weight * spr_loss

    # Update Network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def save_model(model, path="default.pth"):
    torch.save(model.state_dict(), path)


if __name__ == "__main__":

    # Env parameters
    model_name = "DRQN_SPR"
    env_name = "CartPole-v1"
    seed = 1
    exp_num = "SEED" + "_" + str(seed)
    suffix = "eps_06"

    # Set gym environment
    env = gym.make(env_name)

    if torch.cuda.is_available():
        device = torch.device("cuda:1")

    # Set the seed
    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)
    env.seed(seed)

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter(
        "runs/" + env_name + "_" + model_name + "_" + exp_num + "_" + suffix
    )

    # Set parameters
    args = SimpleNamespace()
    args.hidden_space = 64
    args.tran1_dim = 256
    args.tran2_dim = 128
    args.action_space = 1 if env_name == "CartPole-v1" else env.action_space.n
    args.projection_out_dim = 16
    args.prediction_dim = 32
    args.grad_norm_clip = 10
    args.batch_size = 64
    args.lr = 5e-5
    args.tau = 1e-2
    args.spr_weight = 0.01
    args.k_steps = 2
    args.device = torch.device("cuda:1")

    buffer_len = int(100000)
    min_epi_num = 64  # Start moment to train the Q network
    episodes = 650
    print_per_iter = 20
    target_update_period = 4
    eps_start = 0.6
    eps_end = 0.001
    eps_decay = 0.995
    tau = 1e-2
    max_step = 2000

    # DRQN param
    random_update = True  # If you want to do random update instead of sequential update
    lookup_step = 15  # If you want to do random update instead of sequential update
    max_epi_len = 128
    max_epi_step = max_step

    # projection,
    # soft_projection,
    # predictor,

    # Create Q functions
    Q = Q_net(
        args,
        state_space=env.observation_space.shape[0],
        action_space=env.action_space.n,
    ).to(device)
    Q_target = Q_net(
        args,
        state_space=env.observation_space.shape[0],
        action_space=env.action_space.n,
    ).to(device)
    #! NOTE: To remove, since they also use soft updates
    # soft_target = Q_net(
    #     state_space=env.observation_space.shape[0], action_space=env.action_space.n
    # ).to(
    #     device
    # )  # for soft-copying in spr training

    freeze_params(Q_target)
    Q_target.load_state_dict(Q.state_dict())

    # Projection networks
    projector = ProjectionHead(args)
    soft_projector = ProjectionHead(args)
    freeze_params(soft_projector)

    # Predictor
    predictor = PredictionHead(args)

    # Transition model
    transition_model = TransitionModel(args.action_space, args)

    # Set optimizer
    score = 0
    score_sum = 0
    params = list(Q.parameters())
    params += projector.parameters()
    params += predictor.parameters()
    params += transition_model.parameters()
    optimizer = optim.Adam(params, lr=args.lr)

    epsilon = eps_start

    episode_memory = EpisodeMemory(
        random_update=random_update,
        max_epi_num=100,
        max_epi_len=600,
        batch_size=args.batch_size,
        lookup_step=lookup_step,
    )

    # Train
    for i in range(episodes):
        losses = []
        s = env.reset()
        obs = s  # Use only Position of Cart and Pole
        done = False

        episode_record = EpisodeBuffer()
        h = Q.init_hidden_state(batch_size=args.batch_size, training=False)

        for t in range(max_step):

            # Get action
            a, h = Q.sample_action(
                torch.from_numpy(obs).float().to(device).unsqueeze(0).unsqueeze(0),
                h.to(device),
                epsilon,
            )

            # Do action
            s_prime, r, done, _ = env.step(a)
            obs_prime = s_prime

            # make data
            done_mask = 0.0 if done else 1.0

            episode_record.put([obs, a, r / 100.0, obs_prime, done_mask])

            obs = obs_prime

            score += r
            score_sum += r

            if len(episode_memory) >= min_epi_num:
                # collecting loss for logging
                loss = train(
                    Q,
                    Q_target,
                    projector,
                    soft_projector,
                    predictor,
                    transition_model,
                    args,
                    episode_memory,
                    device,
                    optimizer=optimizer,
                    batch_size=args.batch_size,
                )

                losses.append(loss.item())

                if (t + 1) % target_update_period == 0:
                    # Q_target.load_state_dict(Q.state_dict()) <- navie update
                    for target_param, local_param in zip(
                        Q_target.parameters(), Q.parameters()
                    ):  # <- soft update
                        target_param.data.copy_(
                            tau * local_param.data + (1.0 - tau) * target_param.data
                        )

            if done:
                break

        episode_memory.put(episode_record)

        epsilon = max(eps_end, epsilon * eps_decay)  # Linear annealing

        if i % print_per_iter == 0 and i != 0:
            print(
                "n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                    i, score_sum / print_per_iter, len(episode_memory), epsilon * 100
                )
            )
            score_sum = 0.0
            save_model(Q, model_name + "_" + exp_num + ".pth")

        # Log the reward
        writer.add_scalar("Rewards per episodes", score, i)
        writer.add_scalar("Loss per episode", np.mean(losses), i)
        score = 0

    writer.close()
    env.close()
