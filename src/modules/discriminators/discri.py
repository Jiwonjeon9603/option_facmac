import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import log_softmax

class Discrimination(nn.Module):
    def __init__(self, scheme, args):
        super(Discrimination, self).__init__()
        self.args = args
        input_shape = self._get_input_shape(scheme)
        
        if self.args.use_layer_norm:
            self.feature_norm = nn.LayerNorm(input_shape)
            self.fc1 = nn.Sequential(nn.Linear(input_shape, args.rnn_hidden_dim), nn.ReLU(), nn.LayerNorm(args.rnn_hidden_dim))
            self.fc_h = nn.Sequential(nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim), nn.ReLU(), nn.LayerNorm(args.rnn_hidden_dim))
        else:
            self.fc1 = nn.Sequential(nn.Linear(input_shape, args.rnn_hidden_dim), nn.ReLU())
            self.fc_h = nn.Sequential(nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim), nn.ReLU())

        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
    
    def init_hidden(self, batch_size):
        self.hidden_states = None

    def forward(self, inputs, hidden_state=None):
        if self.args.use_layer_norm:
            x = self.feature_norm(inputs)
        else:
            x = inputs
        x = self.fc1(x)
        x = self.fc_h(x)
        x = self.fc2(x)
        log_x = log_softmax(x, dim=-1)
        return log_x
    

    def _get_input_shape(self, scheme):
        # observation
        input = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input += self.args.n_agents
        input += scheme["options_onehot"]["vshape"][0]
        return input


class MIOpt(nn.Module):
    def __init__(self, scheme, args):
        super(MIOpt, self).__init__()
        self.args = args

        input_shape = self._get_input_shape(scheme) + scheme["options_onehot"]["vshape"][0]
        
        if self.args.use_layer_norm:
            self.feature_norm = nn.LayerNorm(input_shape)
            self.fc1 = nn.Sequential(nn.Linear(input_shape, args.rnn_hidden_dim), nn.ReLU(), nn.LayerNorm(args.rnn_hidden_dim))
            self.fc_h = nn.Sequential(nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim), nn.ReLU(), nn.LayerNorm(args.rnn_hidden_dim))
        else:
            self.fc1 = nn.Sequential(nn.Linear(input_shape, args.rnn_hidden_dim), nn.ReLU())
            self.fc_h = nn.Sequential(nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim), nn.ReLU())

        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_options)
    
    def init_hidden(self, batch_size):
        self.hidden_states = None

    def forward(self, inputs, hidden_state=None):
        if self.args.use_layer_norm:
            x = self.feature_norm(inputs)
        else:
            x = inputs
        x = self.fc1(x)
        x = self.fc_h(x)
        x = self.fc2(x)
        log_x = log_softmax(x, dim=-1)
        return log_x

    def _get_input_shape(self, scheme):
        # observation
        input_shape = 2*scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += 2*scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += 2*self.args.n_agents
        return input_shape

