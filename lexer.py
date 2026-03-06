"""
Lexer and parser for L-System files.

Supports:
- Parametric symbols: A(x,y)
- Context-sensitive rules: A < B > C -> D
- Conditional rules: A(x) : x > 5 -> B(x/2)
- Stochastic rules: A : 0.5 -> B, A : 0.5 -> C
"""
import re
from typing import List, Dict, Tuple, Optional, Callable
from symbol import Symbol, ProductionRule, LSystemContext
import numpy as np


class LSystemParser:
    """Parser for L-system definition files."""

    def __init__(self):
        self.context = LSystemContext()
        self.axiom = []
        self.rules = []

    def parse_file(self, filename: str) -> Tuple[List[Symbol], List[ProductionRule], LSystemContext]:
        """Parse an L-system file and return axiom, rules, and context."""
        with open(filename, 'r') as f:
            content = f.read()
        return self.parse(content)

    def parse(self, content: str) -> Tuple[List[Symbol], List[ProductionRule], LSystemContext]:
        """Parse L-system definition from string."""
        lines = content.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Parse variable assignments (e.g., angle = 25.7)
            if '=' in line and '->' not in line:
                self._parse_assignment(line)
            # Parse axiom (e.g., axiom: F(1))
            elif line.startswith('axiom:'):
                self.axiom = self._parse_symbol_string(line.split(':', 1)[1].strip())
            # Parse production rules
            elif '->' in line:
                rule = self._parse_rule(line)
                if rule:
                    self.rules.append(rule)

        return self.axiom, self.rules, self.context

    def _parse_assignment(self, line: str):
        """Parse variable assignment like 'angle = 25.7'."""
        parts = line.split('=')
        if len(parts) == 2:
            var_name = parts[0].strip()
            value_str = parts[1].strip()
            try:
                value = float(value_str)
            except ValueError:
                value = value_str
            self.context.set(var_name, value)

    def _parse_symbol_string(self, s: str) -> List[Symbol]:
        """Parse a string of symbols like 'F(1.5)A(2,3)B'."""
        symbols = []
        i = 0
        while i < len(s):
            char = s[i]

            # Skip whitespace
            if char.isspace():
                i += 1
                continue

            # Check if there are parameters
            if i + 1 < len(s) and s[i + 1] == '(':
                # Find matching closing parenthesis
                j = s.find(')', i + 1)
                if j == -1:
                    raise ValueError(f"Unmatched parenthesis in: {s}")

                params_str = s[i + 2:j]
                params = []
                if params_str:
                    for p in params_str.split(','):
                        p = p.strip()
                        try:
                            params.append(float(p))
                        except ValueError:
                            # Could be a variable reference
                            val = self.context.get(p)
                            if val is not None:
                                params.append(float(val))
                            else:
                                params.append(0.0)

                symbols.append(Symbol(char, params))
                i = j + 1
            else:
                symbols.append(Symbol(char, []))
                i += 1

        return symbols

    def _parse_rule(self, line: str) -> Optional[ProductionRule]:
        """Parse a production rule."""
        # Handle probability prefix (e.g., "A : 0.5 -> B")
        probability = 1.0
        if ':' in line:
            parts = line.split('->')
            if len(parts) == 2:
                left_part = parts[0]
                # Check if there's a probability specification
                if ':' in left_part:
                    prob_parts = left_part.split(':')
                    if len(prob_parts) >= 2:
                        # Try to parse the last part before -> as probability
                        try:
                            # Check if we have a condition or probability
                            prob_candidate = prob_parts[-1].strip()
                            # If it's a number, it's a probability
                            if prob_candidate.replace('.', '').replace('-', '').isdigit():
                                probability = float(prob_candidate)
                                left_part = ':'.join(prob_parts[:-1])
                            else:
                                # It's a condition, keep the whole left part
                                left_part = parts[0]
                        except ValueError:
                            pass

        # Split into predecessor and successor
        if '->' not in line:
            return None

        parts = line.split('->')
        if len(parts) != 2:
            return None

        pred_part = parts[0].strip()
        succ_part = parts[1].strip()

        # Parse context (e.g., "A < B > C" or "B > C" or "A < B")
        left_context = None
        right_context = None
        condition = None
        predecessor = None
        param_names = []

        # First, check if there's a condition (separate it from context checks)
        # Split on ':' to isolate the actual predecessor from the condition
        condition_str = None
        pred_with_context = pred_part
        if ':' in pred_part:
            # Could be either a condition or probability
            colon_parts = pred_part.split(':', 1)
            potential_pred = colon_parts[0].strip()
            potential_cond = colon_parts[1].strip()

            # Check if it's a probability (just a number) or a condition (expression)
            if not potential_cond.replace('.', '').replace('-', '').replace('+', '').replace('e', '').isdigit():
                # It's a condition, not a probability
                pred_with_context = potential_pred
                condition_str = potential_cond

        # Handle context sensitivity (now check only the pred_with_context, not the condition)
        if '<' in pred_with_context or '>' in pred_with_context:
            # Parse context-sensitive rule
            if '<' in pred_with_context and '>' in pred_with_context:
                # Both left and right context
                match = re.match(r'(\w+)\s*<\s*(\w+(?:\([^)]*\))?)\s*>\s*(\w+)', pred_with_context)
                if match:
                    left_context = match.group(1)
                    pred_with_params = match.group(2)
                    right_context = match.group(3)
                    predecessor, param_names = self._parse_predecessor(pred_with_params)
            elif '<' in pred_with_context:
                # Only left context
                match = re.match(r'(\w+)\s*<\s*(\w+(?:\([^)]*\))?)', pred_with_context)
                if match:
                    left_context = match.group(1)
                    pred_with_params = match.group(2)
                    predecessor, param_names = self._parse_predecessor(pred_with_params)
            elif '>' in pred_with_context:
                # Only right context
                match = re.match(r'(\w+(?:\([^)]*\))?)\s*>\s*(\w+)', pred_with_context)
                if match:
                    pred_with_params = match.group(1)
                    right_context = match.group(2)
                    predecessor, param_names = self._parse_predecessor(pred_with_params)
        else:
            # No context, just parse predecessor
            predecessor, param_names = self._parse_predecessor(pred_with_context)

        # Now handle the condition if we extracted one earlier
        if condition_str:
            condition = self._create_condition(condition_str, param_names)

        if predecessor is None:
            return None

        # Create successor generator function
        successor_func = self._create_successor_function(succ_part, param_names)

        return ProductionRule(
            predecessor=predecessor,
            successor=successor_func,
            probability=probability,
            left_context=left_context,
            right_context=right_context,
            condition=condition
        )

    def _parse_predecessor(self, pred_str: str) -> Tuple[str, List[str]]:
        """Parse predecessor and extract parameter names."""
        pred_str = pred_str.strip()

        # Check for parameters
        match = re.match(r'(\w+)\(([^)]*)\)', pred_str)
        if match:
            char = match.group(1)
            params_str = match.group(2)
            param_names = [p.strip() for p in params_str.split(',')] if params_str else []
            return char, param_names
        else:
            return pred_str, []

    def _create_condition(self, condition_str: str, param_names: List[str]) -> Callable:
        """Create a condition function from a string expression."""
        def condition_func(context: dict, params: List[float]) -> bool:
            # Create local scope with parameters
            local_scope = context.copy()
            for i, name in enumerate(param_names):
                if i < len(params):
                    local_scope[name] = params[i]

            try:
                # Safely evaluate the condition
                return bool(eval(condition_str, {"__builtins__": {}}, local_scope))
            except:
                return False

        return condition_func

    def _create_successor_function(self, succ_str: str, param_names: List[str]) -> Callable:
        """Create a successor generator function."""
        def successor_func(context: dict, params: List[float]) -> List[Symbol]:
            # Create local scope with parameters
            local_scope = context.copy()
            for i, name in enumerate(param_names):
                if i < len(params):
                    local_scope[name] = params[i]

            # Add common math functions
            import math
            local_scope.update({
                'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                'sqrt': math.sqrt, 'abs': abs, 'pow': pow,
                'pi': math.pi, 'e': math.e
            })

            symbols = []
            i = 0
            while i < len(succ_str):
                char = succ_str[i]

                # Skip whitespace
                if char.isspace():
                    i += 1
                    continue

                # Check if there are parameters
                if i + 1 < len(succ_str) and succ_str[i + 1] == '(':
                    # Find matching closing parenthesis
                    j = succ_str.find(')', i + 1)
                    if j == -1:
                        i += 1
                        continue

                    params_str = succ_str[i + 2:j]
                    sym_params = []
                    if params_str:
                        for p in params_str.split(','):
                            p = p.strip()
                            try:
                                # Try to evaluate as expression
                                val = eval(p, {"__builtins__": {}}, local_scope)
                                sym_params.append(float(val))
                            except:
                                sym_params.append(0.0)

                    symbols.append(Symbol(char, sym_params))
                    i = j + 1
                else:
                    symbols.append(Symbol(char, []))
                    i += 1

            return symbols

        return successor_func
