# Parametric tree with decreasing branch length
# Demonstrates parametric L-systems

angle = 25.0

axiom: A(10)

A(l) -> F(l)[+A(l*0.7)][-A(l*0.7)]
