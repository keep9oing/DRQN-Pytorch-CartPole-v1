import gym
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace as SN
import pickle
import os
from torch.optim import RMSprop
from tqdm import trange
import matplotlib.pyplot as plt

import numpy as np
from generate_offline_data import SequenceReplayMemory


class ObsEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.fc1 = nn.Linear(args.input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        #         self.fc3 = nn.Linear(128, args.latent_dim)
        self.to(args.device)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)

        return h


def online_init_hidden(batch_size, model):
    """
    Encoder init hidden has batch size 1, so need to expand.
    """
    return model.init_hidden().expand(batch_size, -1)


def offline_init_hidden(batch_size, model):
    return model.init_hidden().expand(batch_size, -1)


class QHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc1 = nn.Linear(args.rnn_hidden_dim, 64)
        self.fc2 = nn.Linear(64, args.n_actions)
        self.to(args.device)

    def forward(self, hidden_encoding):
        # x is the output of fc3 in ObsEncoder -> just "connecting" here??
        x = F.relu(self.fc1(hidden_encoding))
        qvals = self.fc2(x)  # pymarl didnt relu the hidden state

        return qvals


class TransitionModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # binary encoding for actions, so n_actions is 1 the way we encode it
        self.fc1 = nn.Linear(args.rnn_hidden_dim + 1, args.transition_dim)
        self.fc2 = nn.Linear(args.transition_dim, 128)
        self.fc3 = nn.Linear(128, args.rnn_hidden_dim)
        self.to(args.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class ProjectionHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.fc1 = nn.Linear(args.rnn_hidden_dim, 256)
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
    criterion = th.nn.CosineSimilarity(dim=1)
    loss = -criterion(x1, x2).mean()

    return loss


def L1_loss(x1, x2):
    loss = F.smooth_l1_loss(x1, x2)

    return loss


def spr_train(states, actions, t_env, ep_batch, loss_fn):
    """
        Input states and actions are time sliced at time t
        """
    # reset hidden states, they become "irrelevant" when considering a
    # "new" time series
    encoding = online_init_hidden(args.batch_size, online_encoder)
    offline_encoding = offline_init_hidden(args.batch_size, offline_encoder)

    soft_copy(offline_encoder, online_encoder, args.tau)
    freeze_params(offline_encoder)
    soft_copy(offline_projection, online_projection, args.tau)
    freeze_params(offline_projection)

    spr_loss = th.tensor([0.0]).to(args.device)
    k_steps = args.k_steps
    k_steps = min(k_steps, ep_batch["maxlen"] - t_env - 1)  #!: adjust for the +1

    encoding = online_encoder(states, encoding)
    actions = th.unsqueeze(actions, 1)
    #     print(encoding)

    for t_step in range(k_steps):
        #         print(encoding)
        #         print(actions)
        encoding_action = th.cat(
            (encoding, actions), 1
        )  # this is wrong, need to slice actions
        #         print(encoding_action)

        next_encoding = transition_model(encoding_action)
        next_projection = online_projection(next_encoding)
        next_prediction = predictor(next_projection)

        #         print(ep_batch["obs"].shape)
        next_state = ep_batch["obs"][:, t_env + t_step + 1]  # +1 correction is key
        next_state = th.from_numpy(next_state).to(args.device)

        with th.no_grad():
            offline_encoding = offline_encoder(next_state, offline_encoding)
            #             print(target_encoding.shape)
            target_projection = offline_projection(offline_encoding)

        loss_ = loss_fn(next_prediction, target_projection)
        spr_loss += loss_

        encoding = next_encoding

    return spr_loss


if __name__ == "__main__":
    # create env just to access env info
    env = gym.make("CartPole-v0")

    buffer_path = "./buffer/random_agent.pkl"
    with open(buffer_path, "rb") as infile:
        buffer = pickle.load(infile)

    seed = 2023
    th.manual_seed(seed)
    np.random.seed(seed)
    train_steps = 500
    n_games = 250

    args = SN()
    args.input_shape = env.observation_space.shape[0]
    args.n_actions = env.action_space.n
    args.tau = 0.99
    args.latent_dim = 8
    args.transition_dim = 256
    args.projection_out_dim = 16
    args.prediction_dim = 32
    args.rnn_hidden_dim = 64
    args.device = th.device("cuda:2")
    args.save_name = "spr_dqn"
    args.batch_size = 64
    args.k_steps = 4
    args.lr = 5e-5
    args.grad_norm_clip = 10
    args.loss = "cosine"

    online_encoder = ObsEncoder(args)
    transition_model = TransitionModel(args)
    online_projection = ProjectionHead(args)
    predictor = PredictionHead(args)

    params = list()
    params += online_encoder.parameters()
    params += transition_model.parameters()
    params += online_projection.parameters()
    params += predictor.parameters()

    import copy

    offline_encoder = copy.deepcopy(online_encoder)
    offline_projection = copy.deepcopy(online_projection)

    freeze_params(offline_encoder)
    freeze_params(offline_projection)

    optimiser = RMSprop(params=params, lr=args.lr)

    losses = []

    for steps in trange(train_steps):
        ep_batch = buffer.sample_episodes(args.batch_size)

        spr_loss = th.tensor([0.0])
        spr_loss = spr_loss.to(args.device)

        for t in range(ep_batch["maxlen"]):
            states = ep_batch["obs"][:, t]  # state across batches at time t
            actions_debug = ep_batch["action"]
            actions = ep_batch["action"][:, t]

            states = th.from_numpy(states).to(args.device)
            actions = th.from_numpy(actions).to(args.device)
            if args.loss == "cosine":
                spr_loss_ = spr_train(states, actions, t, ep_batch, cosine_loss)
            elif args.loss == "L1":
                spr_loss_ = spr_train(states, actions, t, ep_batch, L1_loss)
            else:
                raise Exception("Loss function must be specified for SPR loss")
            spr_loss += spr_loss_ / args.k_steps

        spr_loss /= args.batch_size
        loss = spr_loss
        losses.append(loss.item())

        optimiser.zero_grad()
        loss.backward()
        grad_norm_spr = th.nn.utils.clip_grad_norm_(params, args.grad_norm_clip)
        optimiser.step()

    plt.plot(np.arange(len(losses)), losses)
    plot_folder = "./plots"
    if not os.path.isdir(plot_folder):
        os.makedirs(plot_folder)
    if not os.path.isdir(os.path.join(plot_folder, "arrays")):
        os.makedirs(os.path.join(plot_folder), "arrays")
    plot_name = f"random_spr_k{args.k_steps}_{args.loss}.png"
    plot_path = os.path.join(plot_folder, plot_name)
    plt.savefig(plot_path)

    array_name = f"random_spr_k{args.k_steps}_{args.loss}.npz"
    array_path = os.path.join(os.path.join(plot_folder, "arrays"), array_name)
    np.save(array_path, losses)
