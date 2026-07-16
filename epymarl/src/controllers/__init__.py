REGISTRY = {}

from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC
from .opt_controller import OptMAC
from .central_basic_controller import CentralBasicMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
REGISTRY["opt_mac"] = OptMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC
