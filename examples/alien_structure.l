# Alien structure with unusual characteristics
# - High pipe exponent (spindly, top-heavy)
# - Negative phototropism (grows away from light)
# - Non-standard gravity

angle = 30.0
gravity = 0.1
phototropism = -0.4
pipe_exponent = 2.8

axiom: F(2)A(15)

# Alien branching pattern
A(r) : r > 1 -> F(r*0.25)[/(45)&(40)A(r*0.35)][\\(45)&(40)A(r*0.35)][^(20)A(r*0.4)]F(r*0.2)
A(r) : r <= 1 -> F(r*0.3)~
