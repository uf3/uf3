import pytest

import uf3.representation.bspline
from uf3.representation.bspline import *
from uf3.data import composition

@pytest.fixture()
def binary_chemistry():
    element_list = ['Ne', 'Xe']
    chemistry_config = composition.ChemicalSystem(element_list)
    yield chemistry_config

@pytest.fixture()
def binary_chemistry():
    element_list = ['Ne', 'Xe']
    chemistry_config = composition.ChemicalSystem(element_list)
    yield chemistry_config

@pytest.fixture
def unary_trio():
    return ('Si','Si','Si')

@pytest.fixture
def binary_sym_trio():
    return ('Si','N','N')

@pytest.fixture
def binary_unsym_trio():
    return ('Si','Si','N') 

@pytest.fixture
def equilateral():
    return dict(r_min = [0.1,0.1,0.1],
                r_max = [8.,8.,8.],
                resolution = [10,10,10])

### The naming convention is TRIANGLE-TYPE_W.R.T_LEG ###
@pytest.fixture
def isosceles_rmax_rjk():
    return dict(r_min = [0.1,0.1,0.1],
                r_max = [5.,5.,10.],
                resolution = [6,6,6])
@pytest.fixture
def isosceles_rmax_rij():
    return dict(r_min = [0.1,0.1,0.1],
                r_max = [5.,10.,5.],
                resolution = [6,6,6])
@pytest.fixture
def isosceles_rmax_rik():
    return dict(r_min = [0.1,0.1,0.1],
                r_max = [10.,5.,5.],
                resolution = [6,6,6])
### Rmin iso ###
@pytest.fixture
def isosceles_rmin_rjk():
    return dict(r_min = [0.1,0.1,0.2],
                r_max = [8.,8.,8.],
                resolution = [6,6,6])
@pytest.fixture
def isosceles_rmin_rij():
    return dict(r_min = [0.2,0.1,0.1],
                r_max = [8.,8.,8.],
                resolution = [6,6,6])
@pytest.fixture
def isosceles_rmin_rik():
    return dict(r_min = [0.1,0.2,0.1],
                r_max = [8.,8.,8.],
                resolution = [6,6,6])

### Resolution iso ###
@pytest.fixture
def isosceles_resolution_rjk():
    return dict(r_min = [0.1,0.1,0.1],
                r_max = [8.,8.,8.],
                resolution = [6,6,12])
@pytest.fixture
def isosceles_resolution_rij():
    return dict(r_min = [0.1,0.1,0.1],
                r_max = [8.,8.,8.],
                resolution = [12,6,6])
@pytest.fixture
def isosceles_resolution_rik():
    return dict(r_min = [0.1,0.1,0.1],
                r_max = [8.,8.,8.],
                resolution = [6,12,6])

@pytest.fixture
def scalene_rmax():
    return dict(r_min = [0.1,0.1,0.1],
                r_max = [4.,5.,10.],
                resolution = [6,6,6])

@pytest.fixture
def scalene_rmin():
    return dict(r_min = [0.1,0.2,0.3],
                r_max = [8.,8.,8.],
                resolution = [6,6,6])

@pytest.fixture
def scalene_resolution():
    return dict(r_min = [0.1,0.1,0.1],
                r_max = [8.,8.,8.],
                resolution = [4,6,12])

@pytest.fixture
def isosceles_rmax_resolution_scalene_rmin():
    return dict(r_min = [0.2,0.1,0.1],
                r_max = [5.,5.,10.],
                resolution = [6,6,12])

@pytest.fixture
def scalene_all_diff():
    return dict(r_min = [0.1,0.1,0.2],
                r_max = [5.,10.,5.],
                resolution = [10,20,20])

class TestFindSym3Body:
    def test_equilateral_unary(self,unary_trio,equilateral):
        assert find_symmetry_3B(unary_trio,**equilateral) == 3
        
    def test_equilateral_binary_sym(self,binary_sym_trio,equilateral):   
        assert find_symmetry_3B(binary_sym_trio,**equilateral) == 2 
        
    def test_equilateral_binary_unsym(self,binary_unsym_trio,equilateral):
        assert find_symmetry_3B(binary_unsym_trio,**equilateral) == 1        
        
    def test_isosceles_unary_rmax_rjk(self,unary_trio,isosceles_rmax_rjk):
        assert find_symmetry_3B(unary_trio,**isosceles_rmax_rjk) == 2
        
    def test_isosceles_binary_sym_rmax_rjk(self,binary_sym_trio,isosceles_rmax_rjk):
        assert find_symmetry_3B(binary_sym_trio,**isosceles_rmax_rjk) == 2
        
    def test_isosceles_binary_unsym_rmax_rjk(self,binary_unsym_trio,isosceles_rmax_rjk):
        assert find_symmetry_3B(binary_unsym_trio,**isosceles_rmax_rjk) == 1
        
    def test_isosceles_unary_rmax_rij(self,unary_trio,isosceles_rmax_rij):
        assert find_symmetry_3B(unary_trio,**isosceles_rmax_rij) == 1
        
    def test_isosceles_binary_sym_rmax_rij(self,binary_sym_trio,isosceles_rmax_rij):
        assert find_symmetry_3B(binary_sym_trio,**isosceles_rmax_rij) == 1
        
    def test_isosceles_binary_unsym_rmax_rij(self,binary_unsym_trio,isosceles_rmax_rij):
        assert find_symmetry_3B(binary_unsym_trio,**isosceles_rmax_rij) == 1        
        
    def test_isosceles_unary_rmax_rik(self,unary_trio,isosceles_rmax_rik):
        assert find_symmetry_3B(unary_trio,**isosceles_rmax_rik) == 1
        
    def test_isosceles_binary_sym_rmax_rik(self,binary_sym_trio,isosceles_rmax_rik):
        assert find_symmetry_3B(binary_sym_trio,**isosceles_rmax_rik) == 1
        
    def test_isosceles_binary_unsym_rmax_rik(self,binary_unsym_trio,isosceles_rmax_rik):
        assert find_symmetry_3B(binary_unsym_trio,**isosceles_rmax_rik) == 1
        
    def test_isosceles_unary_rmin_rjk(self,unary_trio,isosceles_rmin_rjk):
        assert find_symmetry_3B(unary_trio,**isosceles_rmin_rjk) == 2
        
    def test_isosceles_binary_sym_rmin_rjk(self,binary_sym_trio,isosceles_rmin_rjk):
        assert find_symmetry_3B(binary_sym_trio,**isosceles_rmin_rjk) == 2
        
    def test_isosceles_binary_unsym_rmin_rjk(self,binary_unsym_trio,isosceles_rmin_rjk):
        assert find_symmetry_3B(binary_unsym_trio,**isosceles_rmin_rjk) == 1
        
    def test_isosceles_unary_rmin_rij(self,unary_trio,isosceles_rmin_rij):
        assert find_symmetry_3B(unary_trio,**isosceles_rmin_rij) == 1
        
    def test_isosceles_binary_sym_rmin_rij(self,binary_sym_trio,isosceles_rmin_rij):
        assert find_symmetry_3B(binary_sym_trio,**isosceles_rmin_rij) == 1
        
    def test_isosceles_binary_unsym_rmin_rij(self,binary_unsym_trio,isosceles_rmin_rij):
        assert find_symmetry_3B(binary_unsym_trio,**isosceles_rmin_rij) == 1        
        
    def test_isosceles_unary_rmin_rik(self,unary_trio,isosceles_rmin_rik):
        assert find_symmetry_3B(unary_trio,**isosceles_rmin_rik) == 1
        
    def test_isosceles_binary_sym_rmin_rik(self,binary_sym_trio,isosceles_rmin_rik):
        assert find_symmetry_3B(binary_sym_trio,**isosceles_rmin_rik) == 1
        
    def test_isosceles_binary_unsym_rmin_rik(self,binary_unsym_trio,isosceles_rmin_rik):
        assert find_symmetry_3B(binary_unsym_trio,**isosceles_rmin_rik) == 1   

    def test_isosceles_unary_resolution_rjk(self, unary_trio, isosceles_resolution_rjk):
        assert find_symmetry_3B(unary_trio, **isosceles_resolution_rjk) == 2

    def test_isosceles_binary_sym_resolution_rjk(self, binary_sym_trio, isosceles_resolution_rjk):
        assert find_symmetry_3B(binary_sym_trio, **isosceles_resolution_rjk) == 2

    def test_isosceles_binary_unsym_resolution_rjk(self, binary_unsym_trio, isosceles_resolution_rjk):
        assert find_symmetry_3B(binary_unsym_trio, **isosceles_resolution_rjk) == 1

    def test_isosceles_unary_resolution_rij(self, unary_trio, isosceles_resolution_rij):
        assert find_symmetry_3B(unary_trio, **isosceles_resolution_rij) == 1

    def test_isosceles_binary_sym_resolution_rij(self, binary_sym_trio, isosceles_resolution_rij):
        assert find_symmetry_3B(binary_sym_trio, **isosceles_resolution_rij) == 1

    def test_isosceles_binary_unsym_resolution_rij(self, binary_unsym_trio, isosceles_resolution_rij):
        assert find_symmetry_3B(binary_unsym_trio, **isosceles_resolution_rij) == 1

    def test_isosceles_unary_resolution_rik(self, unary_trio, isosceles_resolution_rik):
        assert find_symmetry_3B(unary_trio, **isosceles_resolution_rik) == 1

    def test_isosceles_binary_sym_resolution_rik(self, binary_sym_trio, isosceles_resolution_rik):
        assert find_symmetry_3B(binary_sym_trio, **isosceles_resolution_rik) == 1

    def test_isosceles_binary_unsym_resolution_rik(self, binary_unsym_trio, isosceles_resolution_rik):
        assert find_symmetry_3B(binary_unsym_trio, **isosceles_resolution_rik) == 1
        
    def test_scalene_unary_rmax(self,unary_trio,scalene_rmax):
        assert find_symmetry_3B(unary_trio,**scalene_rmax) == 1
        
    def test_scalene_binary_sym_rmax(self,binary_sym_trio,scalene_rmax):
        assert find_symmetry_3B(binary_sym_trio,**scalene_rmax) == 1
        
    def test_scalene_binary_unsym_rmax(self,binary_unsym_trio,scalene_rmax):
        assert find_symmetry_3B(binary_unsym_trio,**scalene_rmax) == 1  
        
    def test_scalene_unary_rmin(self,unary_trio,scalene_rmin):
        assert find_symmetry_3B(unary_trio,**scalene_rmin) == 1
        
    def test_scalene_binary_sym_rmin(self,binary_sym_trio,scalene_rmin):
        assert find_symmetry_3B(binary_sym_trio,**scalene_rmin) == 1
        
    def test_scalene_binary_unsym_rmin(self,binary_unsym_trio,scalene_rmin):
        assert find_symmetry_3B(binary_unsym_trio,**scalene_rmin) == 1            

    def test_scalene_unary_resolution(self,unary_trio,scalene_resolution):
        assert find_symmetry_3B(unary_trio,**scalene_resolution) == 1
        
    def test_scalene_binary_sym_resolution(self,binary_sym_trio,scalene_resolution):
        assert find_symmetry_3B(binary_sym_trio,**scalene_resolution) == 1
        
    def test_scalene_binary_unsym_resolution(self,binary_unsym_trio,scalene_resolution):
        assert find_symmetry_3B(binary_unsym_trio,**scalene_resolution) == 1   

    def test_binary_sym_isosceles_rmax_resolution_scalene_rmin(self,binary_sym_trio,isosceles_rmax_resolution_scalene_rmin):
        assert find_symmetry_3B(binary_sym_trio,**isosceles_rmax_resolution_scalene_rmin) == 1

    def test_scalene_unary_all_diff(self,unary_trio,scalene_all_diff):
        assert find_symmetry_3B(unary_trio,**scalene_all_diff) == 1

    def test_scalene_binary_sym_all_diff(self,binary_sym_trio,scalene_all_diff):
        assert find_symmetry_3B(binary_sym_trio,**scalene_all_diff) == 1

    def test_scalene_binary_unsym_all_diff(self,binary_unsym_trio,scalene_all_diff):
        assert find_symmetry_3B(binary_unsym_trio,**scalene_all_diff) == 1
       
class TestKnots:
    def test_knot_sequence_from_points(self):
        sequence = knot_sequence_from_points([1, 2, 3])
        assert np.allclose(sequence, [1, 1, 1, 1, 2, 3, 3, 3, 3])

    def test_get_knot_subintervals(self):
        sequence = knot_sequence_from_points([1, 2, 3])
        subintervals = get_knot_subintervals(sequence)
        assert np.allclose(subintervals[0], [1, 1, 1, 1, 2])
        assert np.allclose(subintervals[2], [1, 1, 2, 3, 3])
        assert np.allclose(subintervals[4], [2, 3, 3, 3, 3])

    def test_generate_uniform_knots(self):
        points = generate_uniform_knots(1, 6, 5, sequence=False)
        sequence = generate_uniform_knots(1, 6, 5, sequence=True)
        assert np.allclose(points, [1, 2, 3, 4, 5, 6])
        assert np.allclose(sequence, [1, 1, 1, 1, 2, 3, 4, 5, 6, 6, 6, 6])

    def test_lammps_knots(self):
        points = generate_lammps_knots(0, 1, 2)
        points = np.round(points, 4)
        assert np.allclose(points, [0, 0, 0, 0, 0.7071, 1, 1, 1, 1])


class TestBSplineConfig:
    def test_regularizer_subdivision(self, binary_chemistry):
        bspline_handler = BSplineBasis(binary_chemistry)
        partitions = bspline_handler.get_feature_partition_sizes()
        # default 15 intervals yields 18 basis functions
        assert np.allclose(partitions, [1, 1, 18, 18, 18])

    def test_custom_knots(self):
        element_list = ['Au', 'Ag']
        chemistry = composition.ChemicalSystem(element_list)
        knots_map = {('Ag', 'Au'): [1, 1, 1, 1, 1.1, 1.1, 1.1, 1.1]}
        bspline_handler = BSplineBasis(chemistry,
                                       knots_map=knots_map)
        assert bspline_handler.r_min_map[('Ag', 'Au')] == 1.0
        assert bspline_handler.r_max_map[('Ag', 'Au')] == 1.1
        assert bspline_handler.resolution_map[('Ag', 'Au')] == 1
        assert bspline_handler.r_min_map[('Au', 'Au')] == 1.0
        assert bspline_handler.r_max_map[('Au', 'Au')] == 8.0
        assert bspline_handler.resolution_map[('Au', 'Au')] == 15

    def test_unary(self):
        element_list = ['Au']
        chemistry = composition.ChemicalSystem(element_list)
        bspline_handler = BSplineBasis(chemistry,
                                       r_min_map={('Au', 'Au'): 1.1})
        assert bspline_handler.r_min_map[('Au', 'Au')] == 1.1
        assert bspline_handler.r_max_map[('Au', 'Au')] == 8.0
        assert bspline_handler.resolution_map[('Au', 'Au')] == 15

    def test_binary(self):
        element_list = ['Ne', 'Xe']
        chemistry = composition.ChemicalSystem(element_list)
        bspline_handler = BSplineBasis(chemistry,
                                       resolution_map={('Ne', 'Xe'): 10})
        assert bspline_handler.r_min_map[('Ne', 'Ne')] == 1.0
        assert bspline_handler.r_max_map[('Xe', 'Xe')] == 8.0
        assert bspline_handler.resolution_map[('Ne', 'Xe')] == 10

    def test_regularizer_degree_2(self):
        ridge_map = {1: 4, 2: 0.25}
        curvature_map = {2: 1}
        element_list = ['Ne', 'Xe']
        chemistry = composition.ChemicalSystem(element_list, degree=2)
        resolution_map = {('Ne', 'Xe'): 2,
                          ('Ne', 'Ne'): 3,
                          ('Xe', 'Xe'): 4,
                         }
        bspline_handler = BSplineBasis(chemistry,
                                       resolution_map=resolution_map,
                                       )
        matrix = bspline_handler.get_regularization_matrix(ridge_map,
                                                           curvature_map)
        ref = np.array([
            # 1-body: 'Ne'
            [2, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            # 1-body: 'Xe'
            [0, 2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            # 2-body: ('Ne', 'Ne')
            [0, 0, .5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0, 0,  0, .5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0, 0,  0,  0, .5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0, 0,  0,  0,  0, .5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0, 0,  0,  0,  0,  0, .5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0, 0,  0,  0,  0,  0,  0, .5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0, 0, -1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0, 0,  1, -2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0, 0,  0,  1, -2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0, 0,  0,  0,  1, -2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0, 0,  0,  0,  0,  1, -2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0, 0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            # 2-body: ('Ne', 'Xe')
            [0, 0,  0,  0,  0,  0,  0,  0, .5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0, 0,  0,  0,  0,  0,  0,  0,  0, .5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0, 0,  0,  0,  0,  0,  0,  0,  0,  0, .5,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, .5,  0,  0,  0,  0,  0,  0,  0,  0],
            [0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, .5,  0,  0,  0,  0,  0,  0,  0],
            [0, 0,  0,  0,  0,  0,  0,  0, -1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0, 0,  0,  0,  0,  0,  0,  0,  1, -2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0, 0,  0,  0,  0,  0,  0,  0,  0,  1, -2,  1,  0,  0,  0,  0,  0,  0,  0,  0],
            [0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -2,  1,  0,  0,  0,  0,  0,  0,  0],
            [0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  0,  0,  0],
            # 2-body: ('Xe', 'Xe')
            [0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, .5,  0,  0,  0,  0,  0,  0],
            [0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, .5,  0,  0,  0,  0,  0],
            [0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, .5,  0,  0,  0,  0],
            [0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, .5,  0,  0,  0],
            [0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, .5,  0,  0],
            [0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, .5,  0],
            [0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, .5],
            [0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  1,  0,  0,  0,  0,  0],
            [0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -2,  1,  0,  0,  0,  0],
            [0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -2,  1,  0,  0,  0],
            [0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -2,  1,  0,  0],
            [0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -2,  1,  0],
            [0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -2,  1],
            [0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -1],
        ])
        assert np.allclose(matrix, ref)

    def test_regularizer_degree_3(self):
        element_list = ['Ne']
        chemistry = composition.ChemicalSystem(element_list, degree=3)
        knots_map = {('Ne', 'Ne'): np.array([0, 0, 0, 0, 3, 6, 6, 6, 6]),
                     ('Ne', 'Ne', 'Ne'): [
                         np.array([0, 0, 0, 0, 1, 2, 3, 3, 3, 3]),
                         np.array([0, 0, 0, 0, 1, 2, 3, 3, 3, 3]),
                         np.array([0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6])
                     ]
                     }
        bspline_handler = BSplineBasis(chemistry,
                                       knots_map=knots_map,
                                       leading_trim=0,
                                       trailing_trim=3
                                       )
        # NOTE: if the following 3 assertions fail, BSplineBasis has changed
        # the way it handles 3-body-related things, such as compression and
        # trimming. Then, the regularization matrix in this test should be
        # updated accordingly.
        assert bspline_handler.symmetry[('Ne', 'Ne', 'Ne')] == 2
        assert np.all(bspline_handler.template_mask[('Ne', 'Ne', 'Ne')] == \
                np.array([0, 1, 2, 3, 4,  # no 5 due to triangle restriction
                          9, 10, 11, 12, 13, 14,
                          18, 19, 20, 21, 22, 23,
                          
                          63, 64, 65, 66, 67, 68,
                          72, 73, 74, 75, 76, 77,
                          
                          126, 127, 128, 129, 130, 131,
                         ]))
        assert np.all(bspline_handler.flat_weights[('Ne', 'Ne', 'Ne')] == \
                np.array([0.5, 0.5, 0.5, 0.5, 0.5,
                          1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1,
                          
                          0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                          1, 1, 1, 1, 1, 1,
                          
                          0.5, 0.5, 0.5, 0.5, 0.5, 0.5
                         ]))

        matrix = bspline_handler.get_regularization_matrix(r1=4,
                                                           r2=9,
                                                           r3=25,
                                                           c2=16,
                                                           c3=1)

        # The number of columns in matrix should be:
        # 1-body + 2-body (uncompressed) + 3-body (compressed)
        assert matrix.shape[1] == len(element_list) + \
                                  len(knots_map[('Ne', 'Ne')]) - 4 + \
                                  len(bspline_handler.template_mask[('Ne', 'Ne', 'Ne')])
        ref = np.array([
# 1-body ridge
[ 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 2-body ridge
[ 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 2-body curvature
[ 0,-4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 4,-8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 4,-8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 4,-8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 4,-4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 3-body ridge
[ 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0],
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0],
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
# 3-body curvature
[0, 0, 0, 0, 0, 0,-3, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 1,-4, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 1,-4, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 1,-4, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1,-4, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#
[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,-4, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,-5, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,-5, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,-5, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,-5, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,-5, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,-4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,-5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,-5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,-5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,-5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,-5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],#
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-5, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,-6, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,-6, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,-6, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,-6, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,-6, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],#
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,-5, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,-6, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,-6, 1, 0, 0, 0, 0, 1, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,-6, 1, 0, 0, 0, 0, 1, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,-6, 1, 0, 0, 0, 0, 1, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,-6, 0, 0, 0, 0, 0, 1],#
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,-5, 1, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1,-6, 1, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1,-6, 1, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1,-6, 1, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1,-6, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1,-6],
        ])
        assert np.all(matrix == ref)

def test_fit_spline_1d():
    x = np.linspace(-1, 7, 1000)
    y = np.sin(x) + 0.5 * x
    knot_sequence = uf3.representation.bspline.generate_lammps_knots(0, 6, 5)
    coefficients = fit_spline_1d(x, y, knot_sequence)
    coefficients = np.round(coefficients, 2)
    assert np.allclose(coefficients,
                       [-0.06, 1.59, 2.37, 1.16, 1.23, 1.77, 2.43, 2.71])
    bspline = interpolate.BSpline(t=knot_sequence,
                                  c=coefficients,
                                  k=3,
                                  extrapolate=False)
    yp = bspline(x[(x > 0) & (x < 6)])
    rmse = np.sqrt(np.mean(np.subtract(y[(x > 0) & (x < 6)], yp) ** 2))
    assert rmse < 0.017


def test_distance_bspline():
    points = np.array([1e-10,  # close to 0
                       0.5,
                       1 - 1e-10])  # close to 1
    sequence = uf3.representation.bspline.knot_sequence_from_points([0, 1])
    subintervals = uf3.representation.bspline.get_knot_subintervals(sequence)
    basis_functions = generate_basis_functions(subintervals)
    values_per_spline = evaluate_basis_functions(points,
                                                 basis_functions,
                                                 flatten=False)
    assert len(values_per_spline) == 4
    assert len(values_per_spline[0]) == 3
    assert np.allclose(values_per_spline[0], [1, 0.125, 0])
    assert np.allclose(values_per_spline[1], [0, 0.375, 0])
    assert np.allclose(values_per_spline[2], [0, 0.375, 0])
    assert np.allclose(values_per_spline[3], [0, 0.125, 1])
    value_per_spline = evaluate_basis_functions(points, basis_functions)
    assert len(value_per_spline) == 4
    assert np.allclose(value_per_spline, [1.125, 0.375, 0.375, 1.125])


def test_force_bspline():
    distances = np.array([3, 4, 3, 5, 4, 5])  # three atom molecular triangle
    drij_dR = np.array([[[-1.0, -0.0, -1.0, -0.0, 0.0, 0.0, ],
                         [-0.0, -1.0, 0.0, 0.0, -1.0, -0.0, ],
                         [-0.0, -0.0, 0.0, 0.0, 0.0, 0.0, ]],
                        [[1.0, 0.0, 1.0, 0.6, 0.0, 0.6],
                         [0.0, 0.0, -0.0, -0.8, -0.0, -0.8],
                         [0.0, 0.0, -0.0, -0.0, 0.0, 0.0, ]],
                        [[0.0, 0.0, -0.0, -0.6, -0.0, -0.6],
                         [0.0, 1.0, 0.0, 0.8, 1.0, 0.8],
                         [0.0, 0.0, 0.0, 0.0, -0.0, -0.0, ]]])

    sequence = uf3.representation.bspline.knot_sequence_from_points([2, 6])
    subintervals = uf3.representation.bspline.get_knot_subintervals(sequence)
    basis_functions = generate_basis_functions(subintervals)
    x = featurize_force_2B(basis_functions, distances, drij_dR, sequence)
    assert x.shape == (3, 3, 4)
    assert np.ptp(x[:, 2, :]) == 0  # no z-direction component
    assert np.ptp(np.sum(x, axis=0)) < 1e-10  # forces cancel along atom axis
    assert np.any(np.ptp(x, axis=0) > 0)  # but should not be entirely zero
    assert np.ptp(np.sum(x, axis=2)) < 1e-10  # values cancel across b-splines
    assert np.any(np.ptp(x, axis=2) > 0)  # but should not be entirely zero

def test_bsplinebasis_r_cut():
    element_list = ['Au']
    chemistry = composition.ChemicalSystem(element_list, degree=3)
    r_max_map = {('Au', 'Au'): 5.0,
                 ('Au', 'Au', 'Au'): [5.1, 5.2, 10.3]}
    bspline_handler = BSplineBasis(chemistry,
                                   r_max_map=r_max_map)
    assert bspline_handler.r_cut == 5.2