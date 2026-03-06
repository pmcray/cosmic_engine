# Pre-historic mega-flora with thick trunks
# Low pipe exponent creates unnaturally thick trunks
# Modeled after Lepidodendron

angle = 18.0
gravity = 0.15
phototropism = 0.1
pipe_exponent = 1.5

axiom: F(8)A(20)

# Thick main trunk with sparse branching
A(r) : r > 2 -> F(r*0.4)!((r*0.05))[&(25)+(60)A(r*0.3)][&(25)-(60)A(r*0.3)]F(r*0.3)A(r*0.7)
A(r) : r <= 2 -> F(r*0.5)~
