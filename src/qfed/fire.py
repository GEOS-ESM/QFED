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
    SAVANNA = 'sv'
    GRASSLAND = 'gl'
    AGRICULTURAL = 'ag'
    PEATLAND = 'pt'
    FLARING = 'fl'
    WASTE = 'ws'
    TEMPERATE_FOREST = 'mf'
    BOREAL_FOREST = 'bf'


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
        description='Boreal Forest',
        type=FireType.BOREAL_FOREST,
        vegetation=VegetationCategory.BOREAL_FOREST,
    ),
    Fire(
        description='Temperate Forest',
        type=FireType.TEMPERATE_FOREST,
        vegetation=VegetationCategory.TEMPERATE_FOREST,
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
    Fire(
        description='Agricultural',
        type=FireType.AGRICULTURAL,
        vegetation=VegetationCategory.AGRICULTURAL,
    ),
)

