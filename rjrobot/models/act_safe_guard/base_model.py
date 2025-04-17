from abc import ABCMeta, abstractmethod
from typing import Any, List, Union

import torch



class BaseActSafeGuardModel(metaclass=ABCMeta):
    """
    Abstract base class for all expert tool models.

    This class serves as an interface that all model implementations should inherit from.
    It ensures that necessary methods and properties are implemented by subclasses.
    """

    @abstractmethod
    def predict(self, observation_data: dict, prompt_text: str,actions:torch.Tensor) -> Any:
        """
        Make a prediction based on the input data.

        Args:
            input_data: The input data for the prediction.

        Returns:
            Model prediction results.
        """
        pass

    @abstractmethod
    def load(self, model_path: str) -> None:
        """
        Load model from a path.

        Args:
            model_path: Path to the model file or directory.
        """
        pass