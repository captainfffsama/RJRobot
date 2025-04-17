import random

import torch

from .base_model import BaseActSafeGuardModel
from rjrobot.common import ActionFlag


class FakeActGuard(BaseActSafeGuardModel):
    def predict(self, obervation_data: dict, prompt_text: str,actions:torch.Tensor):
        if random.random() > 0.3:
            return ActionFlag.SAFE
        else:
            return ActionFlag.STOP

    def load(self):
        pass
