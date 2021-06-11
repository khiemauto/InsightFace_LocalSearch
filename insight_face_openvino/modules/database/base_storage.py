import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Union


class BaseFaceStorage(ABC):
    """
    Base class for storing and accessing face descriptors.
    """

    @abstractmethod
    def load(self, path: str) -> None:
        """Load database."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """Save database."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Clears the database."""
        raise NotImplementedError

    @abstractmethod
    def find(self, descriptor: np.ndarray, top_k: int) -> List[Tuple[int, int, np.ndarray]]:
        """Add descriptor with specified user_id."""
        raise NotImplementedError

    @abstractmethod
    def add_descriptor(self, descriptor: np.ndarray, user_id: int) -> Union[int, None]:
        """Add descriptor with specified user_id."""
        raise NotImplementedError

    @abstractmethod
    def remove_descriptor(self, descriptor_id: int) -> None:
        """Updates user id list of descriptor ids."""
        raise NotImplementedError

    @abstractmethod
    def remove_user(self, user_id: int) -> None:
        """Removes user_id and all of it descriptor ids."""
        raise NotImplementedError
