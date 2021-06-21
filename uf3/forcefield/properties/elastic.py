import warnings
import ase
from ase import units
from ase.calculators import calculator as ase_calc
try:
    from elastic import get_elementary_deformations
    from elastic import get_BM_EOS, get_elastic_tensor
    from elastic.elastic import get_bulk_modulus
    USE_ELASTIC = True
except ImportError:
    USE_ELASTIC = False


def get_elastic_constants(geom, calc):
    """
    Args:
        geom (ase.Atoms)
        calc (ase_calc.Calculator)

    Returns:
        elastic_constants (list): list of elastic constants.
            i.e. [**Cij, bulk_modulus]
    """
    if not USE_ELASTIC:
        warnings.warn("elastic could not be imported.", RuntimeWarning)
        return None
    geom.calc = calc
    systems = get_elementary_deformations(geom, n=5, d=2)
    Cij, Bij = get_elastic_tensor(geom, systems)
    get_BM_EOS(geom, systems)
    bulk_modulus = get_bulk_modulus(geom)
    elastic_constants = list(Cij/units.GPa)
    elastic_constants.append(bulk_modulus/units.GPa)
    geom.calc = None
    return elastic_constants
