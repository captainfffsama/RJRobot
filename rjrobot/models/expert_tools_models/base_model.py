from abc import ABCMeta, abstractmethod
from typing import Any, List, Union


class ExpertToolsBaseModel(metaclass=ABCMeta):
    """
    Abstract base class for all expert tool models.
    
    This class serves as an interface that all model implementations should inherit from.
    It ensures that necessary methods and properties are implemented by subclasses.
    """
    
    def __init__(self, inputdata_modality: str,outdata_modality: str):
        """
        Initialize the base model.
        """
        self._inputdata_modality = inputdata_modality
        self._outdata_modality = outdata_modality

    
    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """
        Make a prediction based on the input data.
        
        Args:
            input_data: The input data for the prediction.
        
        Returns:
            Model prediction results.
        """
        pass

    def batch_predict(self,input_datas: List[Any]) -> List[Any]:
        return [self.predict(x) for x in input_datas]
    
    @abstractmethod
    def load(self, model_path: str) -> None:
        """
        Load model from a path.
        
        Args:
            model_path: Path to the model file or directory.
        """
        pass
    
    @property
    def inputdata_modality(self) -> List[str]:
        return self._inputdata_modality
    
    @property
    def outdata_modality(self) -> List[str]:
        return self._outdata_modality
