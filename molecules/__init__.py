from molecules.n2 import n2, n2_zenko
from molecules.h2 import h2
from molecules.lih import lih
from molecules.h2o import h2o

MOLECULES = {
    "h2": {"molecule": h2, "active_electrons": None, "active_orbitals": None},
    "lih": {"molecule": lih, "active_electrons": 2, "active_orbitals": 3},
    "h2o": {"molecule": h2o, "active_electrons": 8, "active_orbitals": 6},
    "n2": {"molecule": n2, "active_electrons": 6, "active_orbitals": 6},
    "n2-large": {"molecule": n2, "active_electrons": 10, "active_orbitals": 8},
    "n2-zenko": {
        "molecule": n2_zenko,
        "active_electrons": 10,
        "active_orbitals": 16,
    },
}