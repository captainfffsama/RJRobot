from abc import ABC, abstractmethod
from typing import Any, List, Union


class BaseModel(ABC):
    """
    Abstract base class for all expert tool models.
    
    This class serves as an interface that all model implementations should inherit from.
    It ensures that necessary methods and properties are implemented by subclasses.
    """
    
    def __init__(self, inputdata_modality: Union[str, List[str]] = None):
        """
        Initialize the base model.
        
        Args:
            inputdata_modality: The modality of input data this model accepts,
                                could be a single string (e.g., 'text', 'image', 'audio')
                                or a list of supported modalities.
        """
        self.inputdata_modality = inputdata_modality or []
    
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
    
    @abstractmethod
    def load(self, model_path: str) -> None:
        """
        Load model from a path.
        
        Args:
            model_path: Path to the model file or directory.
        """
        pass
    
    @property
    def supported_modalities(self) -> List[str]:
        """
        Get the list of input modalities supported by this model.
        
        Returns:
            List of supported modality strings.
        """
        if isinstance(self.inputdata_modality, str):
            return [self.inputdata_modality]
        return self.inputdata_modality
