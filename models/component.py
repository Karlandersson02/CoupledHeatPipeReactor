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

    @abstractmethod
    def post_process(self, X: np.ndarray) -> tuple:
        ...

    @abstractmethod
    def pack(self, X_tuple: tuple) -> np.ndarray:
        ...

    @abstractmethod
    def unpack(self, X) -> tuple:
        ...