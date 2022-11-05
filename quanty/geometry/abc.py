from dataclasses import dataclass


@dataclass(frozen=True)
class Geometry:
    def distance(self, i, j):
        raise NotImplementedError
