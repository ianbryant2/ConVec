from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent
from .central_rnn_agent import CentralRNNAgent, CentralRNNNSAgent


REGISTRY = {}
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["rnn_feat"] = RNNFeatureAgent
REGISTRY["central_rnn"] = CentralRNNAgent
REGISTRY["central_rnn_ns"] = CentralRNNNSAgent
