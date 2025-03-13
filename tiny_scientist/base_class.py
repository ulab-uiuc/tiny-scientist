from typing import Dict, List


class BaseTool(ABC):
    def __init__(self, formatter):
        self.formatter = formatter

    def run(self) -> Dict[str, Dict[str, str]]:
        raise NotImplementedError
    
    
class Writer:
    def __init__(self, tools: List[BaseTool], iter_num: int = 10):
        self.tools = tools
        self.iter_num = iter_num

    def write(self, idea: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:        
        pass

    def rewrite(self, info: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
        pass

    def run(self, idea: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
        self.paper = self.write(idea)
        for i in self.iter_num:
            for tool in self.tools:
                info = tool.run(self.paper)
                self.paper = self.rewrite(info)
        return self.paper
    
    def _xxx():


class TinyScientist:
    def __init__(self, tools: List[BaseTool], output_format: str = 'acl')
        self.writer = Writer(tools, formatter)

    def load(self, data):
        if data is pdf:
            self.input_formatter = PDFFormatter(data)

    def save(self, data):
        if self.output_format is 'acl':
            self.output_formatter = ACLFormatter(data)
        return self.output_formatter.format(data)
    
    def write(self, idea: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
        return self.writer.run(idea)