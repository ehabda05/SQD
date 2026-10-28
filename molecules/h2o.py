from .molecule import Atom, Molecule

h2o = Molecule(
    name="H2O",
    atoms=[
        Atom("O", (0.0, 0.0, 0.0)),
        Atom("H", (0.0, 0.0, 0.957)),
        Atom("H", (0.0, 0.957, 0.0)),
    ],
    charge=0,
    spin=0,
    basis="sto3g",
)
