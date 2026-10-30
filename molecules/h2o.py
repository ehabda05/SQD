from chemistry.molecule import Atom, Molecule

import math


bond_length = 0.958
bond_angle = math.radians(104.5)

h2o = Molecule(
    name="H2O",
    atoms=[
        Atom("O", (0.0, 0.0, 0.0)),
        Atom("H", (0.0, 0.0, bond_length)),
        Atom(
            "H",
            (
                bond_length * math.sin(bond_angle),
                0.0,
                bond_length * math.cos(bond_angle),
            ),
        ),
    ],
    charge=0,
    spin=0,
    basis="sto3g",
)
