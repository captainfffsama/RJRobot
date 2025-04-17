from .base_model import EncoderBaseModel


class FakeDirectPassEncoder(EncoderBaseModel):
    def __init__(self, inputdata_modality):
        self._inputdata_modality = inputdata_modality

    def predict(self, *args):
        return args

    def load(self, *args):
        pass
