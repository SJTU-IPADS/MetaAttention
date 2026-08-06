from .arch_base import Arch
from .A100 import *
from .RTX4090 import *
from .H100 import *
from .MI250 import *

import torch

AttnDevice = {
    (8,0): A100,
    (8,9): RTX4090,
    (9,0): H100,
}

AttnDeviceAMD = {
    (9,0): MI250,
}

def get_attn_device():
    if torch.version.cuda is not None:
        AttnDeviceDict = AttnDevice
    elif torch.version.hip is not None:
        AttnDeviceDict = AttnDeviceAMD
    else:
        raise RuntimeError("Unsupported device type")
    
    current_device = torch.cuda.current_device()
    device_cap = torch.cuda.get_device_capability(current_device)
    
    if device_cap in AttnDeviceDict:
        return AttnDeviceDict[device_cap]()
    else:
        print(f"Warning: Unsupported device capability {device_cap}, defaulting to H100 settings.")
        return H100()

