import numpy as np
import torch
import torch.nn as nn
import operator
from functools import reduce

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400
HIDDEN3_UNITS = 400

import logging

log = logging.getLogger("root")


class PENN(nn.Module):
    """
    (P)robabilistic (E)nsemble of (N)eural (N)etworks
    """

    def __init__(self, num_nets, state_dim, action_dim, learning_rate, device=None):
        """
        :param num_nets: number of networks in the ensemble
        :param state_dim: state dimension
        :param action_dim: action dimension
        :param learning_rate:
        """

        super().__init__()
        self.num_nets = num_nets
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # Log variance bounds
        self.max_logvar = torch.tensor(
            -3 * np.ones([1, self.state_dim]), dtype=torch.float, device=self.device
        )
        self.min_logvar = torch.tensor(
            -7 * np.ones([1, self.state_dim]), dtype=torch.float, device=self.device
        )

        # Create or load networks
        self.networks = nn.ModuleList(
            [self.create_network(n) for n in range(self.num_nets)]
        ).to(device=self.device)
        self.opt = torch.optim.Adam(self.networks.parameters(), lr=learning_rate)

    def forward(self, inputs):
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, device=self.device, dtype=torch.float)
        return [self.get_output(self.networks[i](inputs)) for i in range(self.num_nets)]

    def get_output(self, output):
        """
        Argument:
          output: the raw output of a single ensemble member
        Return:
          mean and log variance
        """
        mean = output[:, 0 : self.state_dim]
        raw_v = output[:, self.state_dim :]
        logvar = self.max_logvar - nn.functional.softplus(self.max_logvar - raw_v)
        logvar = self.min_logvar + nn.functional.softplus(logvar - self.min_logvar)
        return mean, logvar

    def get_loss(self, targ, mean, logvar):
        # TODO: write your code here
        inv_var = torch.exp(-logvar)
        mse = (mean - targ) ** 2
        loss = torch.mean(torch.sum(0.5 * (logvar + mse * inv_var), dim=1))
        return loss

        raise NotImplementedError

    def create_network(self, n):
        layer_sizes = [
            self.state_dim + self.action_dim,
            HIDDEN1_UNITS,
            HIDDEN2_UNITS,
            HIDDEN3_UNITS,
        ]
        layers = reduce(
            operator.add,
            [
                [nn.Linear(a, b), nn.ReLU()]
                for a, b in zip(layer_sizes[0:-1], layer_sizes[1:])
            ],
        )
        layers += [nn.Linear(layer_sizes[-1], 2 * self.state_dim)]
        return nn.Sequential(*layers)

    def train_model(self, inputs, targets, batch_size=128, num_train_itrs=5):
        """
        Training the Probabilistic Ensemble (Algorithm 2)
        Argument:
          inputs: state and action inputs. Assumes that inputs are standardized.
          targets: resulting states
        Return:
            List containing the average loss of all the networks at each train iteration

        """
        # TODO: write your code here

        inputs_t  = inputs if torch.is_tensor(inputs)  else torch.tensor(inputs,  dtype=torch.float, device=self.device)
        targets_t = targets if torch.is_tensor(targets) else torch.tensor(targets, dtype=torch.float, device=self.device)

        N = inputs_t.shape[0]
        avg_losses = []

        for _ in range(num_train_itrs):
            losses_this_itr = []

            for net in self.networks:
                # Sample with replacement for bootstrap-style training
                idx = np.random.choice(N, size=batch_size, replace=True)
                batch_inp  = inputs_t[idx]
                batch_targ = targets_t[idx]

                mean, logvar = self.get_output(net(batch_inp))
                loss = self.get_loss(batch_targ, mean, logvar)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                losses_this_itr.append(loss.item())

            avg_losses.append(float(np.mean(losses_this_itr)))

        return avg_losses

        # raise NotImplementedError
