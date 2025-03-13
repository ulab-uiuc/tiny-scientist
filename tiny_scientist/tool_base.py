import abc
from typing import Dict


class BaseTool(abc.ABC):

    @abc.abstractmethod
    def run(self, query: str) -> Dict[str, Dict[str, str]]:
        pass
