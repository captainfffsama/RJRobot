# -*- coding: utf-8 -*-

# @Description:
# @Author: CaptainHu
# @Date: 2025-04-15 15:44:37
# @LastEditors: CaptainHu
from typing import Any
import torch

from .base_model import ExpertToolsBaseModel


class FakeDetectModel(ExpertToolsBaseModel):
    def __init__(
        self, inputdata_modality: str = "images", outputdata_modality: str = "bbox"
    ):
        super().__init__(inputdata_modality, outputdata_modality)

    def predict(self, input_data: Any) -> Any:
        return torch.tensor([[200, 200, 400, 400]])

    def load(self, model_path: str) -> None:
        pass


class FakeSegModel(ExpertToolsBaseModel):
    def __init__(
        self, inputdata_modality: str = "images", outputdata_modality: str = "images"
    ):
        super().__init__(inputdata_modality, outputdata_modality)

    def predict(self, input_data: Any) -> Any:
        return torch.zeros([1, *input_data.shape[1:]])

    def load(self, model_path: str) -> None:
        pass
