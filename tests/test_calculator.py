import numpy as np
import ase
from uf3.data import composition
from uf3.representation import bspline
from uf3.regression import least_squares
from uf3.forcefield import calculator


class TestCalculator:
    def test_unary(self):
        element_list = ['W']
        chemical_system = composition.ChemicalSystem(element_list)
        bspline_config = bspline.BSplineBasis(chemical_system,
                                              r_min_map={('W', 'W'): 2.0},
                                              r_max_map={('W', 'W'): 6.0},
                                              resolution_map={('W', 'W'): 20},
                                              knot_strategy='lammps')
        model = least_squares.WeightedLinearModel(
            bspline_config=bspline_config)
        pair = bspline_config.interactions_map[2][0]
        x = np.linspace(bspline_config.r_min_map[pair],
                        bspline_config.r_max_map[pair],
                        1000)
        y = 4 * 0.87 * ((2.5 / x)**12 - (2.5 / x)**6)
        knot_sequence = bspline_config.knots_map[pair]
        coefficient_vector = bspline.fit_spline_1d(x,
                                                   y,
                                                   knot_sequence)
        coefficient_vector = np.insert(coefficient_vector, 0, 0)
        model.coefficients = coefficient_vector
        calc = calculator.UFCalculator(model)
        assert len(calc.solutions) == 2
        assert len(calc.pair_potentials) == 1
        geom = ase.Atoms('W2',
                          positions=[[0, 0, 0], [1.5, 1.5, 1.5]],
                          pbc=False,
                          cell=None)
        energy = calc.get_potential_energy(geom)
        assert np.isclose(energy, -1.21578)
        geom.calc = calc
        forces = geom.get_forces()
        assert np.allclose(forces, [[-3.96244881, -3.96244881, -3.96244881],
                                    [3.96244881, 3.96244881, 3.96244881]])
        geom.set_pbc([True, True, True])
        geom.set_cell([[3, 0, 0], [3, 5, 0], [0, 0, 3]])
        assert np.isclose(geom.get_potential_energy(), -15.33335)
        forces = geom.get_forces()
        assert np.allclose(forces, [[0, -17.3656864, 0], [0, 17.3656864, 0]])
