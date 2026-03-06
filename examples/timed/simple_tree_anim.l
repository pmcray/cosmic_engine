# Simple 3D Tree for Animation
# Demonstrates smooth growth with branching

angle: 22.5
length: 1.0
width: 0.12

axiom: A

rules:
A -> F[&+A][&-A]/[^+A][^-A]
