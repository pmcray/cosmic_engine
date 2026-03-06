# Resource-limited tree growth
# Growth depends on available energy from photosynthesis

angle = 25.7
gravity = 0.2
phototropism = 0.3
pipe_exponent = 2.0

axiom: A(10)

# Grow only if resource > 0.5
A(r) : r > 0.5 -> F(r*0.3)[+(30)~A(r*0.4)][-(30)~A(r*0.4)]F(r*0.2)A(r*0.6)

# Become dormant if resources too low
A(r) : r <= 0.5 -> F(r*0.5)~
