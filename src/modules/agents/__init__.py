REGISTRY = {}

from .mlp_agent import MLPAgent
from .rnn_agent import RNNAgent
from .comix_agent import CEMAgent, CEMRecurrentAgent
from .qmix_agent import QMIXRNNAgent, FFAgent, OptQMIXRNNAgent, ActQMIXRNNAgent

REGISTRY["mlp"] = MLPAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["cem"] = CEMAgent
REGISTRY["cemrnn"] = CEMRecurrentAgent
REGISTRY["qmixrnn"] = QMIXRNNAgent
REGISTRY["ff"] = FFAgent
REGISTRY["opt_qmixrnn"] = OptQMIXRNNAgent
REGISTRY["act_qmixrnn"] = ActQMIXRNNAgent