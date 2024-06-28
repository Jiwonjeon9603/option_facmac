from .basic_controller import BasicMAC
from .cqmix_controller import CQMixMAC
from .opt_controller import OptMAC
from .act_controller import ActMAC

REGISTRY = {}
REGISTRY["basic_mac"] = BasicMAC
REGISTRY["cqmix_mac"] = CQMixMAC
REGISTRY["opt_mac"] = OptMAC
REGISTRY["act_mac"] = ActMAC