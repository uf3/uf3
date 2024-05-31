import numpy as np
import ase
import os
import uf3
from uf3.data import composition
from uf3.representation import bspline
from uf3.regression import least_squares
from uf3.forcefield import calculator


class TestCalculator:
    def test_unary_dimer(self):
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

    def test_unary_trimer(self):
        geom = ase.Atoms("W3",
                         positions=[[0, 0, 0], [2, 0, 0], [0, 3, 0]],
                         pbc=False,
                         cell=None)
        pkg_directory = os.path.dirname(os.path.dirname(uf3.__file__))
        data_directory = os.path.join(pkg_directory, "tests/data")
        model_file = os.path.join(data_directory, "precalculated_ref",
                                  "model_unary.json")
        model = least_squares.WeightedLinearModel.from_json(model_file)
        calc = calculator.UFCalculator(model)
        geom.calc = calc
        energy = geom.get_potential_energy()
        assert np.isclose(energy, -18.79979353611411)
        forces = geom.get_forces()
        assert np.allclose(forces, [[-12.26367499,   0.15140673,   0.        ],
                                    [ 12.05608935,   0.31137845,   0.        ],
                                    [  0.20758563,  -0.46278518,   0.        ]]
                           )

    def test_unary_pbc(self):
        geom = ase.Atoms("W8",
                         positions=[[ 0.00,  0.00,  0.00], [ 2.89,  0.12, -0.04],
                                    [-0.32,  2.71, -0.11], [ 2.65,  2.81,  0.37],
                                    [ 0.00,  0.00,  3.00], [ 2.64,  0.00,  3.00],
                                    [-0.08,  2.94,  3.16], [ 2.53,  2.87,  3.23]],
                         pbc=True,
                         cell=np.eye(3)*2.74*2)
        pkg_directory = os.path.dirname(os.path.dirname(uf3.__file__))
        data_directory = os.path.join(pkg_directory, "tests/data")
        model_file = os.path.join(data_directory, "precalculated_ref",
                                  "model_unary.json")
        model = least_squares.WeightedLinearModel.from_json(model_file)
        calc = calculator.UFCalculator(model)
        geom.calc = calc
        energy = geom.get_potential_energy()
        assert np.isclose(energy, -76.358888229785)
        forces = geom.get_forces()
        assert np.allclose(forces, [[ 1.36696442, -0.46307   ,  1.78573347],
                                    [ 0.20112587,  0.17014795,  1.22172728],
                                    [-0.66043959, -1.08374173,  6.78845939],
                                    [-1.30913745,  0.36888897,  1.48182124],
                                    [-0.33315563,  1.28359885, -1.56572912],
                                    [ 0.01504262,  0.06574851, -2.38044283],
                                    [ 0.25436762,  0.2491558 , -7.48063062],
                                    [ 0.46523214, -0.59072835,  0.14906119]]
                           )

    def test_binary(self):
        geom = ase.Atoms("NeXe", positions=[[0, 0, 0], [3.1, 0, 0]], pbc=False)
        pkg_directory = os.path.dirname(os.path.dirname(uf3.__file__))
        data_directory = os.path.join(pkg_directory, "tests/data")
        model_file = os.path.join(data_directory, "precalculated_ref",
                                  "model_binary.json")
        model = least_squares.WeightedLinearModel.from_json(model_file)
        calc = calculator.UFCalculator(model)
        geom.calc = calc
        energy = geom.get_potential_energy()
        assert np.isclose(energy, 0.3464031387757268)
        forces = geom.get_forces()
        assert np.allclose(forces, [[-0.28138023,  0.        ,  0.        ],
                                    [ 0.28138023,  0.        ,  0.        ]]
                           )
    