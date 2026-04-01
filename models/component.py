from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class Component(ABC):

    @abstractmethod
    def assemble(self):
        ...
    
    @abstractmethod
    def initial_guess(self) -> np.ndarray:
        ...
    
    @abstractmethod
    def get_residuals(self, X: np.ndarray) -> np.ndarray:
        ...