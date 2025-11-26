from abc import ABC, abstractmethod
import pandas as pd 
from typing import Optional

class IDataLoader(ABC):
    """Responsibility: load dataset and provide optional caching."""
    @abstractmethod
    def load(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def preview(self, n: int = 5) -> pd.DataFrame:
        """Return a small sample without loading entire dataset if possible."""
        pass


class CSVLoader(IDataLoader):
    """Loads CSV and supports sampling (preview) without full load."""
    def __init__(self, file_path: str, read_kwargs: dict = None):
        self.file_path = file_path
        self.read_kwargs = read_kwargs or {}
        self._df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(self.file_path, **self.read_kwargs)
        return self._df

    def preview(self, n: int = 5) -> pd.DataFrame:
        # If already loaded, return head
        if self._df is not None:
            return self._df.head(n)
        # Otherwise read only n rows
        return pd.read_csv(self.file_path, nrows=n, **self.read_kwargs)
