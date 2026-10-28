from .molecule import Atom, Molecule

h2 = Molecule(
    name="H2",
    atoms=[
        Atom("H", (0.0, 0.0, 0.0)),
        Atom("H", (0.0, 0.0, 0.735)),
    ],
    charge=0,
    spin=0,
    basis="sto3g",
)
