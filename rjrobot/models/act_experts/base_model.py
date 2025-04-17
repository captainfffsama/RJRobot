from abc import ABCMeta, abstractmethod
from typing import Any, List, Union


class ActionBaseModel(metaclass=ABCMeta):
    
    @abstractmethod
    def predict(self, observation_data: Any, prompt_text: str) -> Any:
        """
        Make a prediction based on the observation data and prompt text.
            observation_data (Any): The observation data to process, typically 
                                    containing images or sensor information.
            prompt_text (str): The text prompt to guide the model's prediction.
            Any: The prediction results from the model, which could be text responses,
                    classifications, or other outputs depending on the model type.
        """
        pass

        
        

