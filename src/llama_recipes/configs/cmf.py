# Copyright (c) j-pong.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass, field
from typing import List

@dataclass
class cmf_config:
     alpha: float=0.99
     q: float=0.001
     gamma: float=0.99
     inference_mode: bool = True