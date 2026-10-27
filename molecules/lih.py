from molecule import Atom, Molecule

lih = Molecule(
    name="LiH",
    atoms=[
        Atom("Li", (0.0, 0.0, 0.0)),
        Atom("H", (0.0, 0.0, 1.6)),
    ],
    charge=0,
    spin=0,
    basis="sto3g",
)
