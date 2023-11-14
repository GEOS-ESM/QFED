'''
Satellites and instruments
'''


from enum import Enum, IntEnum, unique
from dataclasses import dataclass

from qfed.vegetation import VegetationCategory


@unique
class FireType(Enum):
    """
    Fire types:
      - broad types of open landscape-scale vegetation fires
      - belowground biomass (duff, peat, etc.) fires
      - crop residue burning and pasture maintenance fires
      - open burning of waste
      - gas flaring
    """

    TROPICAL_FOREST = 'tf'
    EXTRATROPICAL_FOREST = 'xf'
    SAVANNA = 'sv'
    GRASSLAND = 'gl'
    AGRICULTURAL = 'ag'
    PEETLAND = 'pt'
    FLARING = 'fl'
    WASTE = 'ws'


@dataclass(frozen=True, eq=True)
class Fire:
    """
    A basic class describing a Fire.
    """

    description: str
    type: FireType
    vegetation: VegetationCategory


BIOMASS_BURNING = (
    Fire(
        description='Tropical Forest',
        type=FireType.TROPICAL_FOREST,
        vegetation=VegetationCategory.TROPICAL_FOREST,
    ),
    Fire(
        description='Extra-tropical Forest',
        type=FireType.EXTRATROPICAL_FOREST,
        vegetation=VegetationCategory.EXTRATROPICAL_FOREST,
    ),
    Fire(
        description='Savanna',
        type=FireType.SAVANNA,
        vegetation=VegetationCategory.SAVANNA,
    ),
    Fire(
        description='Grassland',
        type=FireType.GRASSLAND,
        vegetation=VegetationCategory.GRASSLAND,
    ),
)

