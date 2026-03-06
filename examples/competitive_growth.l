# Plant optimized for competitive light capture
# Maximizes leaf area in well-lit zones

angle = 20.0
gravity = 0.25
phototropism = 0.6
pipe_exponent = 2.0

axiom: F(2)A(12)

# Aggressive lateral branching toward light
A(r) : r > 1.5 -> F(r*0.2)[+(25)~A(r*0.35)][-(25)~A(r*0.35)][+(70)~A(r*0.25)][-(70)~A(r*0.25)]F(r*0.15)A(r*0.5)

# Simple growth in low resource
A(r) : r <= 1.5 -> F(r*0.4)~A(r*0.8)
