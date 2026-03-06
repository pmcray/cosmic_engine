# Parametric Tree Animation
# Uses parameters to control branch length reduction

angle: 25
length: 1.0
width: 0.1

axiom: A(10)

rules:
A(x) : x > 1 -> F(x)[+A(x*0.7)][-A(x*0.7)]
A(x) : x <= 1 -> F(x)
