import torch


class FakeActExpert(object):
    def predict(self, observation_data: dict, prompt_text: str):
        return torch.tensor(
            [[0.0291, -1.7709, 1.9263, -1.8545, -1.6152, 0.0393, 0.0157]]
        )
