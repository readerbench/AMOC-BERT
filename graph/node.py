from enum import Enum


class AmocNodeType(Enum):
    TEXT_BASED = 1
    INFERRED = 2


class AmocNode:
    
    def __init__(self, text: str, node_type: AmocNodeType, active: bool = False):
        self.text = text
        self.node_type = node_type
        self.active = active
        
    def __str__(self) -> str:
        return self.text + " (" + self.node_type.name + ")"
    
    def __repr__(self) -> str:
        return self.text + " (" + self.node_type.name + ")"
    
    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, AmocNode):
            return self.text == __o.text
        else:
            return False
    
    def __hash__(self) -> int:
        return hash(self.text)
    

