import random

import torch

from .base_model import BaseActSafeGuardModel, ActionFlag


class FakeActGuard(BaseActSafeGuardModel):
    def predict(self, inputs, **kwds):
        if random.random() > 0.5:
            return ActionFlag.SAFE
        else:
            return ActionFlag.STOP

    def load(self):
        pass
