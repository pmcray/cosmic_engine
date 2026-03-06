# Conditional parametric L-system
# Branch only if parameter is large enough

angle = 25.7

axiom: A(20)

A(x) : x > 1 -> F(x)[+A(x*0.6)][-A(x*0.7)]A(x*0.8)
A(x) : x <= 1 -> F(x)
