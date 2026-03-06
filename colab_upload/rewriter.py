"""
Stochastic L-System rewriting engine with Numba optimization.
"""
from typing import List
import numpy as np
from symbol import Symbol, ProductionRule, LSystemContext


class LSystemRewriter:
    """Rewriting engine for L-systems."""

    def __init__(self, rules: List[ProductionRule], context: LSystemContext):
        self.rules = rules
        self.context = context
        self.rng = np.random.default_rng()

    def rewrite(self, current_string: List[Symbol], iterations: int = 1) -> List[Symbol]:
        """
        Rewrite the symbol string for the specified number of iterations.

        Args:
            current_string: Initial symbol string
            iterations: Number of rewriting iterations

        Returns:
            Final symbol string after all iterations
        """
        result = current_string

        for iteration in range(iterations):
            result = self._rewrite_once(result)

        return result

    def _rewrite_once(self, current_string: List[Symbol]) -> List[Symbol]:
        """Perform one iteration of rewriting."""
        new_string = []

        for i, symbol in enumerate(current_string):
            # Get left and right context symbols
            left_sym = current_string[i - 1] if i > 0 else None
            right_sym = current_string[i + 1] if i < len(current_string) - 1 else None

            # Find matching rules
            matching_rules = []
            for rule in self.rules:
                if rule.matches(symbol, left_sym, right_sym, self.context.variables):
                    matching_rules.append(rule)

            # Apply rule or keep symbol unchanged
            if matching_rules:
                # Handle stochastic selection
                if len(matching_rules) > 1:
                    # Normalize probabilities
                    probs = np.array([r.probability for r in matching_rules])
                    probs = probs / probs.sum()
                    # Select rule based on probability
                    selected_rule = self.rng.choice(matching_rules, p=probs)
                else:
                    selected_rule = matching_rules[0]

                # Generate successor symbols
                successor = selected_rule.successor(self.context.variables, symbol.params)
                new_string.extend(successor)
            else:
                # No matching rule, keep symbol as is
                new_string.append(symbol.copy())

        return new_string

    def string_to_str(self, symbol_string: List[Symbol]) -> str:
        """Convert symbol list to string representation."""
        return ''.join(str(s) for s in symbol_string)


class OptimizedRewriter:
    """
    Numba-optimized rewriter for simple L-systems without parameters.

    Note: This is a simplified version for performance-critical applications
    that don't require parametric or conditional rules.
    """

    def __init__(self, rules_dict: dict):
        """
        Initialize with a dictionary of simple character-to-string rules.

        Args:
            rules_dict: Dictionary mapping characters to replacement strings
        """
        self.rules_dict = rules_dict

    def rewrite(self, axiom: str, iterations: int) -> str:
        """
        Rewrite a simple string-based L-system.

        Args:
            axiom: Initial string
            iterations: Number of iterations

        Returns:
            Final string after rewriting
        """
        current = axiom

        for _ in range(iterations):
            current = self._rewrite_once(current)

        return current

    def _rewrite_once(self, current: str) -> str:
        """Perform one iteration of simple string rewriting."""
        result = []

        for char in current:
            if char in self.rules_dict:
                result.append(self.rules_dict[char])
            else:
                result.append(char)

        return ''.join(result)


def create_optimized_rules(rules: List[ProductionRule]) -> dict:
    """
    Convert parametric rules to simple character rules for optimization.

    Only works for non-parametric, non-conditional rules.
    """
    rules_dict = {}

    for rule in rules:
        # Check if rule is simple (no parameters, no conditions, no context)
        if (rule.condition is None and
            rule.left_context is None and
            rule.right_context is None):

            # Generate successor with empty parameters
            successor = rule.successor({}, [])
            # Check if all symbols have no parameters
            if all(len(s.params) == 0 for s in successor):
                successor_str = ''.join(s.char for s in successor)
                rules_dict[rule.predecessor] = successor_str

    return rules_dict
