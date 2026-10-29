from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Atom:
    symbol: str
    position: Tuple[float, float, float]


@dataclass
class Molecule:
    name: str
    atoms: List[Atom]
    charge: int = 0
    spin: int = 0
    basis: str = "sto3g"

    def to_pyscf_string(self) -> str:
        return "; ".join(
            f"{atom.symbol} "
            f"{atom.position[0]} "
            f"{atom.position[1]} "
            f"{atom.position[2]}"
            for atom in self.atoms
        )