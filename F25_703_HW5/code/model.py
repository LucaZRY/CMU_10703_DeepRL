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

        const = 0.5 * np.log(2* np.pi)

        loss = torch.mean(torch.sum(0.5 * (logvar + mse * inv_var) + const, dim=1))
        return loss

        # raise NotImplementedError

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



        perm = np.random.permutation(N)
        val_n = max( int(0.1 * N), batch_size )  # at least one batch
        val_idx = torch.as_tensor(perm[:val_n], device=self.device)
        tr_idx  = torch.as_tensor(perm[val_n:], device=self.device)
        Xtr, Ytr = inputs_t[tr_idx],  targets_t[tr_idx]
        Xva, Yva = inputs_t[val_idx], targets_t[val_idx]

        avg_val_losses = []

        for _ in range(num_train_itrs):
            # ---- training step for each net (bootstrap + SGD) ----
            for net in self.networks:
                idx = torch.randint(0, Xtr.shape[0], (batch_size,), device=self.device)
                batch_inp  = Xtr[idx]
                batch_targ = Ytr[idx]

                mean, logvar = self.get_output(net(batch_inp))
                loss = self.get_loss(batch_targ, mean, logvar)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            # ---- evaluate mean NLL on the *validation* split (smooth & stable) ----
            with torch.no_grad():
                val_losses_each_net = []
                # iterate in reasonably large chunks to reduce variance further
                bs_eval = 2048
                for net in self.networks:
                    total, count = 0.0, 0
                    for start in range(0, Xva.shape[0], bs_eval):
                        end = min(start + bs_eval, Xva.shape[0])
                        mean, logvar = self.get_output(net(Xva[start:end]))
                        l = self.get_loss(Yva[start:end], mean, logvar)
                        total += l.item() * (end - start)
                        count += (end - start)
                    val_losses_each_net.append(total / max(count, 1))
                avg_val_losses.append(float(np.mean(val_losses_each_net)))

        return avg_val_losses

        # avg_losses = []

        # for _ in range(num_train_itrs):
        #     losses_this_itr = []

        #     for net in self.networks:
        #         # Sample with replacement for bootstrap-style training
        #         idx = np.random.choice(N, size=batch_size, replace=True)
        #         batch_inp  = inputs_t[idx]
        #         batch_targ = targets_t[idx]

        #         mean, logvar = self.get_output(net(batch_inp))
        #         loss = self.get_loss(batch_targ, mean, logvar)

        #         self.opt.zero_grad()
        #         loss.backward()
        #         self.opt.step()

        #         losses_this_itr.append(loss.item())

        #     avg_losses.append(float(np.mean(losses_this_itr)))

        # return avg_losses

        # raise NotImplementedError
