# Stochastic plant with random branching
# Demonstrates stochastic L-systems

angle = 22.5

axiom: F

F : 0.33 -> F[+F]F[-F]F
F : 0.33 -> F[+F]F
F : 0.34 -> F[-F]F
