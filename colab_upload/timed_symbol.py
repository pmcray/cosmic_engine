"""
Timed Symbol System for Developmental L-Systems.

Implements DOL-systems (Developmental L-systems) where each symbol has an age
and terminal age, as described in ABOP Chapter 1, Section 1.10.1.
"""
import numpy as np
from typing import List, Optional, Callable
from dataclasses import dataclass, field
from symbol import Symbol as BaseSymbol


@dataclass
class TimedSymbol:
    """Symbol with age for developmental L-systems."""

    char: str
    age: float = 0.0
    terminal_age: float = 1.0
    params: List[float] = field(default_factory=list)

    # Growth properties
    max_length: float = 1.0
    max_width: float = 0.1
    growth_rate: float = 1.0  # Units per time

    # State
    is_mature: bool = False

    def __post_init__(self):
        """Initialize symbol state."""
        if self.params is None:
            self.params = []

        self.is_mature = self.age >= self.terminal_age

    def advance_age(self, dt: float):
        """
        Advance the age of this symbol.

        Args:
            dt: Time delta
        """
        self.age += dt
        self.is_mature = self.age >= self.terminal_age

    def get_growth_factor(self) -> float:
        """
        Calculate growth factor (0.0 to 1.0) based on age.

        Uses smooth interpolation to prevent "popping" of new branches.

        Returns:
            Growth factor from 0.0 (just born) to 1.0 (fully mature)
        """
        if self.terminal_age <= 0:
            return 1.0

        # Linear growth
        factor = min(1.0, self.age / self.terminal_age)

        # Smooth S-curve for more natural growth
        # factor = self._smooth_step(factor)

        return factor

    def _smooth_step(self, t: float) -> float:
        """Smoothstep interpolation for natural growth curves."""
        if t <= 0:
            return 0.0
        if t >= 1:
            return 1.0
        return t * t * (3.0 - 2.0 * t)

    def _smooth_step_cubic(self, t: float) -> float:
        """Cubic smoothstep for even smoother transitions."""
        if t <= 0:
            return 0.0
        if t >= 1:
            return 1.0
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    def get_current_length(self) -> float:
        """Get current length based on growth factor."""
        return self.max_length * self.get_growth_factor()

    def get_current_width(self) -> float:
        """Get current width based on growth factor."""
        return self.max_width * self.get_growth_factor()

    def to_base_symbol(self) -> BaseSymbol:
        """Convert to base Symbol for use with existing systems."""
        return BaseSymbol(self.char, self.params.copy())

    def copy(self):
        """Create a copy of this symbol."""
        return TimedSymbol(
            char=self.char,
            age=self.age,
            terminal_age=self.terminal_age,
            params=self.params.copy(),
            max_length=self.max_length,
            max_width=self.max_width,
            growth_rate=self.growth_rate,
            is_mature=self.is_mature
        )

    def __repr__(self):
        if self.params:
            params_str = ','.join(str(p) for p in self.params)
            return f"{self.char}({params_str})@{self.age:.2f}"
        return f"{self.char}@{self.age:.2f}"


class GrowthFunction:
    """Defines how a symbol grows over time."""

    def __init__(self, function_type: str = 'linear', **kwargs):
        """
        Initialize growth function.

        Args:
            function_type: 'linear', 'exponential', 'sigmoid', 'custom'
            **kwargs: Additional parameters for the function
        """
        self.function_type = function_type
        self.params = kwargs

    def evaluate(self, age: float, terminal_age: float) -> float:
        """
        Evaluate growth function at given age.

        Args:
            age: Current age
            terminal_age: Terminal age

        Returns:
            Growth factor (0.0 to 1.0)
        """
        if terminal_age <= 0:
            return 1.0

        t = min(1.0, age / terminal_age)

        if self.function_type == 'linear':
            return t

        elif self.function_type == 'exponential':
            rate = self.params.get('rate', 2.0)
            return (np.exp(rate * t) - 1) / (np.exp(rate) - 1)

        elif self.function_type == 'sigmoid':
            # S-curve (logistic function)
            k = self.params.get('steepness', 10.0)
            return 1.0 / (1.0 + np.exp(-k * (t - 0.5)))

        elif self.function_type == 'smoothstep':
            return t * t * (3.0 - 2.0 * t)

        elif self.function_type == 'cubic':
            return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

        elif self.function_type == 'custom':
            func = self.params.get('function')
            if func:
                return func(t)
            return t

        else:
            return t


class TimedString:
    """A string of timed symbols representing the current L-system state."""

    def __init__(self, symbols: Optional[List[TimedSymbol]] = None):
        self.symbols = symbols if symbols else []

    def __len__(self):
        return len(self.symbols)

    def __getitem__(self, index):
        return self.symbols[index]

    def __iter__(self):
        return iter(self.symbols)

    def append(self, symbol: TimedSymbol):
        """Add a symbol to the string."""
        self.symbols.append(symbol)

    def extend(self, symbols: List[TimedSymbol]):
        """Add multiple symbols to the string."""
        self.symbols.extend(symbols)

    def advance_all(self, dt: float):
        """Advance age of all symbols by dt."""
        for symbol in self.symbols:
            symbol.advance_age(dt)

    def get_mature_symbols(self) -> List[TimedSymbol]:
        """Get all symbols that have reached maturity."""
        return [s for s in self.symbols if s.is_mature]

    def get_immature_symbols(self) -> List[TimedSymbol]:
        """Get all symbols that haven't reached maturity."""
        return [s for s in self.symbols if not s.is_mature]

    def to_base_symbols(self) -> List[BaseSymbol]:
        """Convert all symbols to base Symbol type."""
        return [s.to_base_symbol() for s in self.symbols]

    def copy(self):
        """Create a deep copy of this string."""
        return TimedString([s.copy() for s in self.symbols])

    def __repr__(self):
        return ''.join(str(s) for s in self.symbols[:20]) + ('...' if len(self.symbols) > 20 else '')


def interpolate_symbol_states(sym1: TimedSymbol, sym2: TimedSymbol, t: float) -> TimedSymbol:
    """
    Interpolate between two symbol states.

    Args:
        sym1: First symbol state
        sym2: Second symbol state
        t: Interpolation factor (0.0 to 1.0)

    Returns:
        Interpolated symbol
    """
    result = sym1.copy()

    # Interpolate age
    result.age = sym1.age + t * (sym2.age - sym1.age)

    # Interpolate params
    if len(sym1.params) == len(sym2.params):
        result.params = [
            p1 + t * (p2 - p1)
            for p1, p2 in zip(sym1.params, sym2.params)
        ]

    # Interpolate growth properties
    result.max_length = sym1.max_length + t * (sym2.max_length - sym1.max_length)
    result.max_width = sym1.max_width + t * (sym2.max_width - sym1.max_width)

    return result


def create_timed_axiom(base_axiom: List[BaseSymbol],
                      default_terminal_age: float = 1.0,
                      default_max_length: float = 1.0) -> TimedString:
    """
    Convert base axiom to timed axiom.

    Args:
        base_axiom: List of base symbols
        default_terminal_age: Default terminal age for all symbols
        default_max_length: Default maximum length

    Returns:
        TimedString
    """
    timed_symbols = []

    for sym in base_axiom:
        timed_sym = TimedSymbol(
            char=sym.char,
            age=0.0,
            terminal_age=default_terminal_age,
            params=sym.params.copy() if sym.params else [],
            max_length=default_max_length
        )
        timed_symbols.append(timed_sym)

    return TimedString(timed_symbols)
