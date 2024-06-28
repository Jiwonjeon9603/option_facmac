import torch as th
import torch.nn as nn
import torch.nn.functional as F


class FACMACCritic(nn.Module):
    def __init__(self, scheme, args):
        super(FACMACCritic, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme) + self.n_actions
        self.output_type = "q"
        self.hidden_states = None

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, 1)

    def init_hidden(self, batch_size):
        # make hidden states on same device as model
        self.hidden_states = None

    def forward(self, inputs, actions, hidden_state=None):
        if actions is not None:
            inputs = th.cat([inputs.view(-1, self.input_shape - self.n_actions),
                             actions.contiguous().view(-1, self.n_actions)], dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q, hidden_state

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        return input_shape


class ActFACMACDiscreteCritic(nn.Module):
    def __init__(self, scheme, args):
        super(ActFACMACDiscreteCritic, self).__init__()
        self.args = args
        self.n_actions = scheme["actions_onehot"]["vshape"][0]
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme) + self.n_actions
        self.output_type = "q"
        self.hidden_states = None

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, 1)

    def init_hidden(self, batch_size):
        # make hidden states on same device as model
        self.hidden_states = None

    def forward(self, inputs, actions, hidden_state=None):
        if actions is not None:
            inputs = th.cat([inputs.reshape(-1, self.input_shape - self.n_actions),
                             actions.contiguous().view(-1, self.n_actions)], dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q, hidden_state

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        # if self.args.obs_last_action:
        #     input_shape += scheme["actions_onehot"]["vshape"][0]
        # if self.args.obs_agent_id:
        #     input_shape += self.args.n_agents
        input_shape += scheme["options_onehot"]["vshape"][0]
        return input_shape


class OptFACMACDiscreteCritic(nn.Module):
    def __init__(self, scheme, args):
        super(OptFACMACDiscreteCritic, self).__init__()
        self.args = args
        self.n_actions = scheme["options_onehot"]["vshape"][0]
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme) + self.args.n_options
        self.output_type = "q"
        self.hidden_states = None

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, 1)

    def init_hidden(self, batch_size):
        # make hidden states on same device as model
        self.hidden_states = None

    def forward(self, inputs, actions, hidden_state=None):
        if actions is not None:
            inputs = th.cat([inputs.reshape(-1, self.input_shape - self.n_actions),
                             actions.contiguous().view(-1, self.n_actions)], dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q, hidden_state

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        # if self.args.obs_last_action:
        #     input_shape += scheme["actions_onehot"]["vshape"][0]
        # if self.args.obs_agent_id:
        #     input_shape += self.args.n_agents
        return input_shape

