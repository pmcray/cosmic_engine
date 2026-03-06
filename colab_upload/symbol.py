"""
Data models for L-System Core.
"""
from typing import List, Optional, Callable, Tuple
from dataclasses import dataclass


@dataclass
class Symbol:
    """A symbol with optional parameters."""
    char: str
    params: List[float] = None

    def __post_init__(self):
        if self.params is None:
            self.params = []

    def __repr__(self):
        if self.params:
            params_str = ','.join(str(p) for p in self.params)
            return f"{self.char}({params_str})"
        return self.char

    def copy(self):
        """Create a copy of this symbol."""
        return Symbol(self.char, self.params.copy() if self.params else [])


@dataclass
class ProductionRule:
    """A production rule for L-system rewriting."""
    predecessor: str
    successor: Callable[[dict, List[float]], List[Symbol]]  # Function that generates symbols
    probability: float = 1.0
    left_context: Optional[str] = None
    right_context: Optional[str] = None
    condition: Optional[Callable[[dict, List[float]], bool]] = None

    def matches(self, symbol: Symbol, left_sym: Optional[Symbol], right_sym: Optional[Symbol], context: dict) -> bool:
        """Check if this rule matches the given symbol and context."""
        # Check predecessor
        if symbol.char != self.predecessor:
            return False

        # Check left context
        if self.left_context is not None:
            if left_sym is None or left_sym.char != self.left_context:
                return False

        # Check right context
        if self.right_context is not None:
            if right_sym is None or right_sym.char != self.right_context:
                return False

        # Check condition
        if self.condition is not None:
            if not self.condition(context, symbol.params):
                return False

        return True


@dataclass
class LSystemContext:
    """Global context for L-system evaluation."""
    variables: dict = None

    def __post_init__(self):
        if self.variables is None:
            self.variables = {}

    def set(self, key: str, value):
        """Set a context variable."""
        self.variables[key] = value

    def get(self, key: str, default=None):
        """Get a context variable."""
        return self.variables.get(key, default)
