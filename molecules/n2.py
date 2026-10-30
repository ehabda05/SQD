from chemistry.molecule import Atom, Molecule

n2 = Molecule(
    name="N2",
    atoms=[
        Atom("N", (0.0, 0.0, 0.0)),
        Atom("N", (0.0, 0.0, 1.09)),
    ],
    charge=0,
    spin=0,
    basis="sto3g",
)

n2_zenko = Molecule(
    name="N2-ZenKO",
    atoms=[
        Atom("N", (0.0, 0.0, 0.0)),
        Atom("N", (0.0, 0.0, 1.0)),
    ],
    charge=0,
    spin=0,
    basis="6-31g",
)