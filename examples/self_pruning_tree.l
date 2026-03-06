# Tree with aggressive self-pruning
# Shaded branches die back to redirect resources

angle = 24.0
gravity = 0.3
phototropism = 0.4
pipe_exponent = 2.0

axiom: F(4)A(15)

# Only grow with good light (r > 2)
A(r) : r > 2 -> F(r*0.3)[&(30)+(45)~A(r*0.45)][&(30)-(45)~A(r*0.45)]F(r*0.2)A(r*0.6)

# Marginal growth
A(r) : r > 0.5 -> F(r*0.5)~A(r*0.7)

# Die back if too shaded
A(r) : r <= 0.5 -> D
