import pytest
import ase
import json
import uf3
from uf3.representation.process import *
from uf3.representation import bspline
from uf3.data import composition
from uf3.data import io
import numpy as np

@pytest.fixture()
def simple_molecule():
    geom = ase.Atoms('Ar3',
                     positions=[[0, 0, 0], [3, 0.0, 0.0], [0, 4.0, 0]],
                     pbc=False,
                     cell=None)
    yield geom


@pytest.fixture()
def simple_water():
    geom = ase.Atoms('H2O',
                     positions=[[0, 0, 0], [3, 0.0, 0.0], [0, 4.0, 0]],
                     pbc=False,
                     cell=None)
    yield geom

@pytest.fixture()
def strained_H2O_molecule():
    geom = ase.Atoms('H2O',
                     positions=[[0, 0, 0], [1.5, 0.0, 0.0], [0, 2.0, 0]],
                     pbc=False,
                     cell=None)
    yield geom

@pytest.fixture()
def simple_molecule_CPtC():
    geom = ase.Atoms('CPtC',
                      positions = [[0., 0., 0.], [0., 1.5, 0.], [0., 0., 2.]],
                      pbc = True,
                      cell = [30.0, 30.0, 30.0])
    yield geom

@pytest.fixture()
def atoms_molecule_CCPt():
    geom = ase.Atoms('C2Pt',
                      positions = [[0., 0., 0.], [0., 0., 2.], [0., 1.5, 0.]],
                      pbc = True,
                      cell = [30.0, 30.0, 30.0])
    yield geom
    
@pytest.fixture()    
def atoms_molecule_Yb2La2():
    geom = ase.Atoms('Yb2La2',
                      positions = [[0., 0., 0.], [0., 0., 2.], [0., 1.5, 0.],[2.,0,0]],
                      pbc = True,
                      cell = [30.0, 30.0, 30.0]) 
    yield geom

@pytest.fixture()
def methane_structure():
    geom = ase.Atoms("CH4",
            positions=[[15.000000000, 15.000000000, 15.000010729],
                [15.629117489, 15.629117489, 15.629128218],
                [14.370881617, 14.370881617, 15.629128218],
                [15.629117489, 14.370881617, 14.370892346],
                [14.370881617, 15.629117489, 14.370892346]],
            pbc=True,
            cell=[30, 30, 30])
    yield geom

@pytest.fixture()
def rattled_steel():
    geom = ase.Atoms('Fe8C3',
            positions=[[ 1.99342831e-01,  7.23471398e-02,  2.29537708e-01],
                       [ 3.27460597e+00,  3.16932506e-03, -9.68273914e-02],
                       [ 3.65842563e-01,  3.07348695e+00, -1.43894877e-01],
                       [ 3.02851201e+00,  2.85731646e+00,  6.85404929e-03],
                       [-1.60754569e-03, -3.82656049e-01,  2.57501643e+00],
                       [ 2.80754249e+00, -3.02566224e-01,  2.88284947e+00],
                       [-8.16048151e-02,  2.53753926e+00,  3.26312975e+00],
                       [ 2.92484474e+00,  2.93350564e+00,  2.58505036e+00],
                       [ 1.32612346e+00,  1.45718452e+00, -1.80198715e-01],
                       [ 1.51013960e+00, -7.01277380e-02,  1.37666125e+00],
                       [-7.03413224e-02,  1.80545564e+00,  1.43230056e+00]],
            pbc=True,
            cell=[5.74, 5.74, 5.74])
    yield geom
    

@pytest.fixture()
def unary_chemistry():
    element_list = ['Ar']
    pair = ('Ar', 'Ar')
    chemistry_config = composition.ChemicalSystem(element_list)
    yield chemistry_config


@pytest.fixture()
def water_chemistry():
    element_list = ['H', 'O']
    chemistry_config = composition.ChemicalSystem(element_list)
    yield chemistry_config


@pytest.fixture()
def binary_chemistry():
    element_list = ['Ne', 'Xe']
    chemistry_config = composition.ChemicalSystem(element_list)
    yield chemistry_config
    
@pytest.fixture()
def binary_chemistry_equal_electronegativity():
    element_list = ['Yb', 'La']
    chemistry_config_1 = composition.ChemicalSystem(element_list,degree=3)
    element_list = ['La', 'Yb']
    chemistry_config_2 = composition.ChemicalSystem(element_list,degree=3)    
    yield [chemistry_config_1,chemistry_config_2]
    
@pytest.fixture()
def binary_chemistry_3B():
    element_list = ['C', 'Pt']
    chemistry_config = composition.ChemicalSystem(element_list,degree=3)
    yield chemistry_config

@pytest.fixture()
def rattled_steel_chemistry():
    element_list = ['Fe', 'C']
    chemistry_config = composition.ChemicalSystem(element_list, degree=3)
    yield chemistry_config

@pytest.fixture()
def methane_chem_config():
    chemistry_config = composition.ChemicalSystem(['H','C'],degree=3)
    yield chemistry_config

@pytest.fixture()
def strained_H2O_molecule_feature():
    # For leading_trim={2: 0, 3: 3} and trailing_trim={2: 3, 3: 3}
    two_body = {('H', 'H'): np.array([0.0, 0.40032798833819255, 1.1900510204081631, 
                    0.40949951409135077, 0.00012147716229348758, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                ('H', 'O'): np.array([0.0, 0.0, 0.20991253644314867, 1.4571185617103986, 
                    1.745019436345967, 0.5846695821185617, 0.0032798833819242057, 
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                ('O', 'O'): np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
    # Selected a subset of three_body from strained_H2O_molecule_feature_old to
    # test the new default leading trim of {2: 0, 3: 3}. See PR #118 for a schematic
    # of what/how these were selected. The numerical values of the selected features
    # are the same as the old.
    three_body = {('H', 'H', 'H'):{"position":np.array([]),"value":np.array([])}, 
                ('H', 'H', 'O'):{"position":np.array([0, 1, 2, 7, 8, 9]),
                    "value":np.array([0.11179061530876638, 0.02854780141611156,
                        5.380932829072594e-05, 0.046232917007898805, 0.00356407243123478,
                        4.6287594228581435e-06])},
                ('H', 'O', 'O'):{"position":np.array([]),"value":np.array([])},
                ('O', 'H', 'H'):{"position":np.array([0, 7, 14]),
                    "value":np.array([0.033415592868540726, 0.03629005247013563,
                                      0.0028744596015948995])},
                    ('O', 'H', 'O'):{"position":np.array([]),"value":np.array([])},
                    ('O', 'O', 'O'):{"position":np.array([]),"value":np.array([])}
                }
    features = {2:two_body,3:three_body}
    yield features

@pytest.fixture()
def strained_H2O_molecule_feature_old():
    # For leading_trim={2: 0, 3: 0} and trailing_trim={2: 3, 3: 3}
    two_body = {('H', 'H'): np.array([0.0, 0.40032798833819255, 1.1900510204081631, 
                    0.40949951409135077, 0.00012147716229348758, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                ('H', 'O'): np.array([0.0, 0.0, 0.20991253644314867, 1.4571185617103986, 
                    1.745019436345967, 0.5846695821185617, 0.0032798833819242057, 
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                ('O', 'O'): np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
    three_body = {('H', 'H', 'H'):{"position":np.array([]),"value":np.array([])}, 
                ('H', 'H', 'O'):{"position":np.array([9, 10, 11, 12, 16, 17, 18, 
                    19, 20, 24, 25, 26, 27, 28, 33, 34, 35, 36, 37, 52, 53, 54, 
                    55, 59, 60, 61, 62, 63, 68, 69, 70, 71, 72, 78, 79, 80, 81, 
                    82, 97, 98, 99, 100, 105, 106, 107, 108, 109, 115, 116, 117, 
                    118, 119, 125, 126, 127, 128, 129, 145, 146, 147, 148, 154, 
                    155, 156, 157, 158, 164, 165, 166, 167, 168, 174, 175, 176, 
                    177, 178]),
                    "value":np.array([8.998308318036238e-06, 5.553122679602926e-05, 
                        2.1162688081307482e-05, 4.16588348057235e-08, 8.998308318036216e-06, 
                        0.0004546957671957677, 0.0022863479443642424, 0.0008419840681063094, 
                        1.652467113960362e-06, 0.0002069610913148332, 0.002806441139065861, 
                        0.005365949534820799, 0.001370294467973358, 2.5828477579548518e-06, 
                        0.0002069610913148332, 0.0022965370010438087, 0.0022191800163791483, 
                        0.00017107547669926987, 2.2218045229719143e-07, 0.0007536083216355335, 
                        0.004650740244167441, 0.0017723751268094983, 3.4889274149793364e-06, 
                        0.0007536083216355316, 0.038080770502645474, 0.1914816403405049, 
                        0.07051616570390326, 0.00013839412079418004, 0.017332991397617244, 
                        0.23503944539676538, 0.44939827354124107, 0.11476216169276852, 
                        0.00021631349972871838, 0.017332991397617244, 0.19233497383741857, 
                        0.1858563263717533, 0.01432757117356382, 1.8607612879889746e-05, 
                        0.0009935632101164991, 0.006131572958728215, 0.0023367134756443622, 
                        4.599829676465292e-06, 0.0009935632101164967, 0.05020599096119924, 
                        0.2524509188568845, 0.09296907418673811, 0.00018245991049978955, 
                        0.022851953832679444, 0.3098778757718547, 0.5924902611364619, 
                        0.15130334750539126, 0.0002851894399408475, 0.022851953832679444, 
                        0.2535759605319199, 0.2450344601418637, 0.018889583885544334, 
                        2.453242494114816e-05, 0.00018746475662575454, 0.0011569005582506068, 
                        0.00044088933502723824, 8.678923917859043e-07, 0.00018746475662575405, 
                        0.00947282848324514, 0.04763224884092161, 0.01754133475221474, 
                        3.442639820750746e-05, 0.004311689402392347, 0.05846752373053862, 
                        0.11179061530876638, 0.02854780141611156, 5.380932829072594e-05, 
                        0.004311689402392347, 0.047844520855079224, 0.046232917007898805, 
                        0.00356407243123478, 4.6287594228581435e-06])},
                ('H', 'O', 'O'):{"position":np.array([]),"value":np.array([])},
                ('O', 'H', 'H'):{"position":np.array([50, 51, 52, 53, 59, 60, 61, 
                    62, 69, 70, 71, 72, 79, 80, 81, 82, 89, 90, 91, 92, 99, 100, 
                    101, 102, 109, 110, 111, 112, 119, 120, 121, 122, 129, 130, 
                    131, 132]),
                    "value":np.array([8.998308318036216e-06, 0.00018699609473419023,
                        0.00016637497150535723, 2.343309457821933e-05, 0.0002069610913148332,
                        0.0043009101788863795, 0.0038266243446232195, 0.0005389611752990452,
                        0.0002069610913148332, 0.0043009101788863795, 0.0038266243446232195,
                        0.0005389611752990452, 0.00035693289661543583, 0.007417511757789529,
                        0.006599540536379156, 0.0009295127516026982, 0.008767351737873276,
                        0.1821965283026791, 0.16210468057005284, 0.02283164515071167,
                        0.008257447599851224, 0.17160008293440834, 0.15267676551808262,
                        0.021503769791279246, 0.01283158766151963, 0.26665643109095494,
                        0.2372507093666391, 0.033415592868540726, 0.01393538014853207,
                        0.28959461871168224, 0.2576593725379629, 0.03629005247013563,
                        0.0011037924870124407, 0.02293818762072729, 0.020408663171323782,
                        0.0028744596015948995])},
                    ('O', 'H', 'O'):{"position":np.array([]),"value":np.array([])},
                    ('O', 'O', 'O'):{"position":np.array([]),"value":np.array([])}
                }
    features = {2:two_body,3:three_body}
    yield features

@pytest.fixture()
def methane_feature():
    # For leading_trim={2: 0, 3: 3} and trailing_trim={2: 3, 3: 3}
    two_body = {('H', 'H'):np.array([0.0, 0.10764117873003697, 4.380510760509621,
                    6.909855011070257, 0.6019930496900838, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                ('H', 'C'):np.array([4.217956715718236, 3.381599561086582, 0.3909862297136271,
                    0.009457493481554552, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0]),
                ('C', 'C'):np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
    # Selected a subset of three_body from methane_feature_old to
    # test the new default leading trim of {2: 0, 3: 3}. See PR #118 for a schematic
    # of what/how these were selected. The numerical values of the selected features
    # are the same as the old.
    three_body = {('H', 'H', 'H'):{"position":np.array([0, 1, 7, 8, 14, 15]),
        "value":np.array([0.6640224780125649, 0.0007053656017778708, 0.01702949612348602, 
            1.8089780359648227e-05, 0.00010918445829116121, 1.159824609519897e-07])},
        ('H', 'H', 'C'):{"position":np.array([0, 14]),
        "value":np.array([1.624998081281485e-06, 2.083732060447781e-08])},
        ('H', 'C', 'C'):{"position":np.array([]),"value":np.array([])},
        ('C', 'H', 'H') :{"position":np.array([0, 1]),
            "value":np.array([8.505596144699058e-07, 9.035168449480808e-10])},
        ('C', 'H', 'C'):{"position":np.array([]),"value":np.array([])},
        ('C', 'C', 'C'):{"position":np.array([]),"value":np.array([])}
        }
    features = {2:two_body,3:three_body}
    yield features

@pytest.fixture()
def methane_feature_old():
    # For leading_trim={2: 0, 3: 0} and trailing_trim={2: 3, 3: 3}
    two_body = {('H', 'H'):np.array([0.0, 0.10764117873003697, 4.380510760509621,
                    6.909855011070257, 0.6019930496900838, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                ('H', 'C'):np.array([4.217956715718236, 3.381599561086582, 0.3909862297136271,
                    0.009457493481554552, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0]),
                ('C', 'C'):np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}

    three_body = {('H', 'H', 'H'):{"position":np.array([43, 44, 45, 46, 51, 52, 53,
        54, 60, 61, 62, 63, 70, 71, 72, 73, 80, 81, 82, 83, 90, 91, 92, 93, 100,
        101, 102, 103, 110, 111, 112, 113, 120, 121, 122, 123, 130, 131, 132, 133]),
        "value":np.array([0.03100005703780889, 0.10614310151474303, 0.040768935952206535,
            4.330727647630547e-05, 0.4045895910301375, 1.385300484452129, 0.532085702407565,
            0.0005652142270837575, 0.25021837004738573, 0.8567388705253303, 0.3290683204255295,
            0.0003495566513918442, 0.0032085455717557163, 0.01098594682978099, 0.004219637039864313,
            4.482358532223911e-06, 1.3201002902211445, 4.519976816290672, 1.7360962954640766,
            0.0018441884856963812, 1.632831640958409, 5.590757927037117, 2.147376971267495,
            0.0022810761679567374, 0.020937770196598957, 0.07169018640060694, 
            0.027535775533769652, 2.9250197881750437e-05, 0.5049122380066092, 
            1.7288016879906885, 0.6640224780125649, 0.0007053656017778708, 
            0.012948960742367371, 0.04433678470046677, 0.01702949612348602, 
            1.8089780359648227e-05, 8.302214310023149e-05, 0.00028426488868429044, 
            0.00010918445829116121, 1.159824609519897e-07])},
        ('H', 'H', 'C'):{"position":np.array([42, 43, 44, 45, 50, 51, 52, 53, 58,
            59, 60, 61, 67, 68, 69, 70, 87, 88, 89, 90, 95, 96, 97, 98, 104, 105, 
            106, 107, 114, 115, 116, 117, 134, 135, 136, 137, 143, 144, 145, 146, 
            153, 154, 155, 156, 163, 164, 165, 166, 183, 184, 185, 186, 193, 194,
            195, 196, 203, 204, 205, 206, 213, 214, 215, 216]),
        "value":np.array([0.8429228766683844, 0.39945724689618667, 0.028847549005400645,
            0.00044543852440743863, 0.48446944868436304, 0.2295878277015724, 
            0.016580112545328155, 0.00025601554105109207, 0.04169744942536497, 
            0.01976022814266378, 0.0014270216753349406, 2.2034815825280747e-05, 
            0.0007619495486119261, 0.00036108436177978177, 2.6076379642926395e-05, 
            4.0264856010784597e-07, 5.500599910594774, 2.606708819255387, 0.18824833193180882, 
            0.0029067654649678636, 3.1614667010247706, 1.498204425914217, 0.10819562276557199, 
            0.001670661814023262, 0.2721019833839744, 0.1289478695660432, 0.009312210543850567, 
            0.0001437909793569346, 0.004972198210514753, 0.0023563017010207734, 0.00017016471554247018, 
            2.6275341375052783e-06, 3.401845164645613, 1.6121186664281806, 0.11642215179136364, 
            0.0017976886525983612, 1.9552085926762723, 0.9265642956744039, 0.06691356617883588, 
            0.0010332205406989095, 0.1682814295732272, 0.07974779000553874, 0.005759135171737034, 
            8.892750896186398e-05, 0.003075055214890228, 0.0014572544228538549, 
            0.00010523834203001602, 1.624998081281485e-06, 0.04362179817874775, 
            0.020672168103937787, 0.0014928791180028496, 2.3051728634462977e-05, 
            0.02507160393820065, 0.01188131697632776, 0.0008580314323557801, 
            1.3248968050901788e-05, 0.002157869686241052, 0.001022604449190199, 
            7.38492847224609e-05, 1.1403158170713884e-06, 3.943137664421636e-05, 
            1.8686346747777566e-05, 1.3494693304707247e-06, 2.083732060447781e-08])},
        ('H', 'C', 'C'):{"position":np.array([]),"value":np.array([])},
        ('C', 'H', 'H') :{"position":np.array([1, 2, 3, 4, 8, 9, 10, 11, 16, 17, 
            18, 19, 24, 25, 26, 27, 43, 44, 45, 46, 51, 52, 53, 54, 60, 61, 62, 
            63, 80, 81, 82, 83, 90, 91, 92, 93, 110, 111, 112, 113]),
            "value":np.array([0.7915185971384979, 2.7101317492548986, 1.0409455360795885, 
                0.0011057565049189097, 0.9098497359452233, 3.115293393422155, 
                1.1965657213353516, 0.0012710658570213517, 0.0783091966940646, 
                0.26812792647763406, 0.10298634678624186, 0.00010939844490385975, 
                0.001430966591375146, 0.0048995791197187265, 0.0018818992894873746, 
                1.999069412209704e-06, 0.2614678116790906, 0.895256561755444, 
                0.34386273724939853, 0.000365272192765014, 0.045008166700019184, 
                0.15410637474653752, 0.0591913448185388, 6.287669460066086e-05, 
                0.0008224472425331765, 0.002816030339823197, 0.0010816205568270109, 
                1.1489640188764265e-06, 0.0019368876198045738, 0.006631834870523742, 
                0.002547248452547226, 2.7058443006122534e-06, 7.078661490949063e-05, 
                0.00024237087186837486, 9.30932148285657e-05, 9.888935039597454e-08, 
                6.467521397549924e-07, 2.214456507004659e-06, 8.505596144699058e-07, 
                9.035168449480808e-10])},
        ('C', 'H', 'C'):{"position":np.array([]),"value":np.array([])},
        ('C', 'C', 'C'):{"position":np.array([]),"value":np.array([])}
        }
    features = {2:two_body,3:three_body}
    yield features



class TestBasis:
    
    def test_equal_electronegativity(self,binary_chemistry_equal_electronegativity, atoms_molecule_Yb2La2):
        bspline_config = bspline.BSplineBasis(binary_chemistry_equal_electronegativity[0])
        bspline_handler = BasisFeaturizer(bspline_config) 
        feature_1 = bspline_handler.featurize_energy_3B(atoms_molecule_Yb2La2)
        
        bspline_config = bspline.BSplineBasis(binary_chemistry_equal_electronegativity[1])
        bspline_handler = BasisFeaturizer(bspline_config) 
        feature_2 = bspline_handler.featurize_energy_3B(atoms_molecule_Yb2La2)  
        assert np.allclose(feature_1,feature_2)
        
    def test_atom_swap_3B(self,binary_chemistry_3B,simple_molecule_CPtC,atoms_molecule_CCPt):
        bspline_config = bspline.BSplineBasis(binary_chemistry_3B)
        bspline_handler = BasisFeaturizer(bspline_config) 
        feature_1 = bspline_handler.featurize_energy_3B(simple_molecule_CPtC)
        feature_1 = feature_1[np.where(feature_1!=0)[0]]
        feature_2 = bspline_handler.featurize_energy_3B(atoms_molecule_CCPt)
        feature_2 = feature_2[np.where(feature_2!=0)[0]]
        assert np.allclose(feature_1,feature_2)
        
    def test_setup(self, unary_chemistry):
        bspline_config = bspline.BSplineBasis(unary_chemistry)
        bspline_handler = BasisFeaturizer(bspline_config)
        assert bspline_handler.r_cut == 8.0
        assert len(bspline_handler.knots_map[('Ar', 'Ar')]) == 22
        assert len(bspline_handler.basis_functions[('Ar', 'Ar')]) == 18
        assert len(bspline_handler.columns) == 20  # 1 + 23 + 1

    def test_energy_features(self, unary_chemistry, simple_molecule):
        bspline_config = bspline.BSplineBasis(unary_chemistry)
        bspline_handler = BasisFeaturizer(bspline_config)
        vector = bspline_handler.featurize_energy_2B(simple_molecule,
                                                     simple_molecule)
        assert len(vector) == 18  # 23 features

    def test_energy_features_strained_H2O_molecule(self, strained_H2O_molecule, \
            strained_H2O_molecule_feature):
        element_list = ['H', 'O']
        chemistry_config = composition.ChemicalSystem(element_list,degree=3)
        bspline_config = bspline.BSplineBasis(chemistry_config)
        bspline_handler = BasisFeaturizer(bspline_config)
        
        features_2B = bspline_handler.featurize_energy_2B(strained_H2O_molecule)
        features_3B = bspline_handler.featurize_energy_3B(strained_H2O_molecule)
        features_con = np.concatenate((features_2B,features_3B))

        features = {}
        features = {key:[] for key in bspline_config.interactions_map[2]}
        features.update({key:[] for key in bspline_config.interactions_map[3]})
        
        for i in features.keys():
            start = bspline_config.get_interaction_partitions()[1][i]-2
            end = bspline_config.get_interaction_partitions()[1][i]-2 + \
                    bspline_config.get_interaction_partitions()[0][i]
            features[i] = features_con[start:end]

        for i in bspline_config.interactions_map[2]:
            assert np.allclose(strained_H2O_molecule_feature[2][i],features[i])

        for i in bspline_config.interactions_map[3]:
            feature = features[i]
            feature_position = np.where(feature!=0)[0]
            feature_value = feature[feature_position]

            assert np.allclose(strained_H2O_molecule_feature[3][i]["position"],feature_position)
            #In my hard-coded features I double counted every interaction,
            #so it needs to be divided by two
            assert np.allclose(strained_H2O_molecule_feature[3][i]["value"]/2,feature_value)

    def test_energy_features_strained_H2O_molecule_old(self, strained_H2O_molecule, \
            strained_H2O_molecule_feature_old):
        element_list = ['H', 'O']
        chemistry_config = composition.ChemicalSystem(element_list,degree=3)
        bspline_config = bspline.BSplineBasis(chemistry_config,
                                              leading_trim={2: 0, 3: 0},
                                              trailing_trim={2: 3, 3: 3})
        bspline_handler = BasisFeaturizer(bspline_config)
        
        features_2B = bspline_handler.featurize_energy_2B(strained_H2O_molecule)
        features_3B = bspline_handler.featurize_energy_3B(strained_H2O_molecule)
        features_con = np.concatenate((features_2B,features_3B))

        features = {}
        features = {key:[] for key in bspline_config.interactions_map[2]}
        features.update({key:[] for key in bspline_config.interactions_map[3]})
        
        for i in features.keys():
            start = bspline_config.get_interaction_partitions()[1][i]-2
            end = bspline_config.get_interaction_partitions()[1][i]-2 + \
                    bspline_config.get_interaction_partitions()[0][i]
            features[i] = features_con[start:end]

        for i in bspline_config.interactions_map[2]:
            assert np.allclose(strained_H2O_molecule_feature_old[2][i],features[i])

        for i in bspline_config.interactions_map[3]:
            feature = features[i]
            feature_position = np.where(feature!=0)[0]
            feature_value = feature[feature_position]

            assert np.allclose(strained_H2O_molecule_feature_old[3][i]["position"],feature_position)
            #In my hard-coded features I double counted every interaction,
            #so it needs to be divided by two
            assert np.allclose(strained_H2O_molecule_feature_old[3][i]["value"]/2,feature_value)

    def test_energy_features_methane(self, methane_chem_config, \
            methane_structure, methane_feature):
        bspline_config = bspline.BSplineBasis(methane_chem_config)
        bspline_handler = BasisFeaturizer(bspline_config)

        features_2B = bspline_handler.featurize_energy_2B(methane_structure)
        features_3B = bspline_handler.featurize_energy_3B(methane_structure)
        features_con = np.concatenate((features_2B,features_3B))

        features = {}
        features = {key:[] for key in bspline_config.interactions_map[2]}
        features.update({key:[] for key in bspline_config.interactions_map[3]})
        
        for i in features.keys():
            start = bspline_config.get_interaction_partitions()[1][i]-2
            end = bspline_config.get_interaction_partitions()[1][i]-2 + \
                    bspline_config.get_interaction_partitions()[0][i]
            features[i] = features_con[start:end]

        for i in bspline_config.interactions_map[2]:
            assert np.allclose(methane_feature[2][i],features[i])

        for i in bspline_config.interactions_map[3]:
            print(i)
            feature = features[i]
            feature_position = np.where(feature!=0)[0]
            feature_value = feature[feature_position]

            assert np.allclose(methane_feature[3][i]["position"],feature_position)
            #In my hard-coded features I double counted every interaction,
            #so it needs to be divided by two
            assert np.allclose(methane_feature[3][i]["value"]/2,feature_value)

    def test_energy_features_methane_old(self, methane_chem_config, \
            methane_structure, methane_feature_old):
        bspline_config = bspline.BSplineBasis(methane_chem_config,
                                              leading_trim={2: 0, 3: 0},
                                              trailing_trim={2: 3, 3: 3})
        bspline_handler = BasisFeaturizer(bspline_config)

        features_2B = bspline_handler.featurize_energy_2B(methane_structure)
        features_3B = bspline_handler.featurize_energy_3B(methane_structure)
        features_con = np.concatenate((features_2B,features_3B))

        features = {}
        features = {key:[] for key in bspline_config.interactions_map[2]}
        features.update({key:[] for key in bspline_config.interactions_map[3]})
        
        for i in features.keys():
            start = bspline_config.get_interaction_partitions()[1][i]-2
            end = bspline_config.get_interaction_partitions()[1][i]-2 + \
                    bspline_config.get_interaction_partitions()[0][i]
            features[i] = features_con[start:end]

        for i in bspline_config.interactions_map[2]:
            assert np.allclose(methane_feature_old[2][i],features[i])

        for i in bspline_config.interactions_map[3]:
            feature = features[i]
            feature_position = np.where(feature!=0)[0]
            feature_value = feature[feature_position]

            assert np.allclose(methane_feature_old[3][i]["position"],feature_position)
            #In my hard-coded features I double counted every interaction,
            #so it needs to be divided by two
            assert np.allclose(methane_feature_old[3][i]["value"]/2,feature_value)


    def test_force_features(self, unary_chemistry, simple_molecule):
        bspline_config = bspline.BSplineBasis(unary_chemistry)
        bspline_handler = BasisFeaturizer(bspline_config)
        vector = bspline_handler.featurize_force_2B(simple_molecule,
                                                    simple_molecule)
        assert vector.shape == (3, 3, 18)  # 3 forces per atom

    def test_evaluate_single(self, unary_chemistry, simple_molecule):
        bspline_config = bspline.BSplineBasis(unary_chemistry)
        bspline_handler = BasisFeaturizer(bspline_config)
        eval_map = bspline_handler.evaluate_configuration(simple_molecule,
                                                          energy=1.5)
        assert len(eval_map['energy']) == 1 + 18 + 1  # number of columns
        assert eval_map['energy'][0] == 1.5  # energy value
        assert eval_map['energy'][1] == 3  # scalar for 1-body energy offset
        eval_map = bspline_handler.evaluate_configuration(simple_molecule,
                                                          name='sample',
                                                          forces=[[2, 0, 0],
                                                                  [0, 0, 0],
                                                                  [0, 0, 0]])
        assert len(eval_map) == 3 + 3 + 3  # 3 atoms, 3 forces each
        assert eval_map[('sample', 'fx_0')][0] == 2  # force value
        assert eval_map[('sample', 'fy_1')][1] == 0  # no 1-body energy offset
        assert len(eval_map[('sample', 'fz_2')]) == 1 + 18 + 1   # columns

    def test_evaluate_unary(self, unary_chemistry, simple_molecule):
        bspline_config = bspline.BSplineBasis(unary_chemistry)
        bspline_handler = BasisFeaturizer(bspline_config)
        df = pd.DataFrame(columns=['geometry', 'energy', 'fx', 'fy', 'fz'])
        df.loc[0] = [None,
                     1.5,
                     [4, 3, 0],
                     [0, 1, 2],
                     [2, 1, 0]]
        df.at[0, 'geometry'] = simple_molecule
        df.loc[1] = [None,
                     1.5,
                     [4.1, 3.1, 0],
                     [0, 1.1, 2.1],
                     [2, 1, 0]]
        df.at[1, 'geometry'] = simple_molecule
        data_coordinator = io.DataCoordinator()
        atoms_key = data_coordinator.atoms_key
        energy_key = data_coordinator.energy_key
        df_features = bspline_handler.evaluate(df,
                                               atoms_key,
                                               energy_key,
                                               progress=False)
        assert len(df_features) == 2 + 6 * 3  # energy and 3 forces per atom
        assert len(df_features.columns) == 1 + 18 + 1
        x, y, w = bspline_handler.get_training_tuples(df_features,
                                                      0.5,
                                                      data_coordinator)
        assert x.shape == (2 + 6 * 3, 18 + 1)
        assert np.allclose(y[:10], [1.5, 4, 3, 0, 0, 1, 2, 2, 1, 0])

    def test_evaluate_binary(self, water_chemistry, simple_water):
        bspline_config = bspline.BSplineBasis(water_chemistry)
        bspline_handler = BasisFeaturizer(bspline_config)
        df = pd.DataFrame(columns=['geometry', 'energy', 'fx', 'fy', 'fz'])
        df.loc[0] = [None,
                     1.5,
                     [4, 3, 0],
                     [0, 1, 2],
                     [2, 1, 0]]
        df.at[0, 'geometry'] = simple_water
        df.loc[1] = [None,
                     1.5,
                     [4.1, 3.1, 0],
                     [0, 1.1, 2.1],
                     [2, 1, 0]]
        df.at[1, 'geometry'] = simple_water
        data_coordinator = io.DataCoordinator()
        print(len(bspline_handler.columns))
        atoms_key = data_coordinator.atoms_key
        energy_key = data_coordinator.energy_key
        df_feats = bspline_handler.evaluate(df,
                                            atoms_key,
                                            energy_key,
                                            progress=False)
        assert len(df_feats) == 2 * (1 + 3 * 3)  # energy and 3 forces per atom
        assert len(df_feats.columns) == 1 + 2 + 18 * 3
        # energy, 23 features per interaction, two 1-body terms
        x, y, w = bspline_handler.get_training_tuples(df_feats,
                                                      0.5,
                                                      data_coordinator)
        assert x.shape == (2 + 6 * 3, 18 * 3 + 2)
        assert np.allclose(y[:10], [1.5, 4, 3, 0, 0, 1, 2, 2, 1, 0])

    def test_evaluate_binary_pbc(self, rattled_steel_chemistry, rattled_steel):
        n_atoms = len(rattled_steel)
        r_min_map = {('Fe', 'Fe'): 0.1, ('Fe', 'C'): 0.1, ('C', 'C'): 0.1,
                     ('Fe', 'Fe', 'Fe'): [1.5, 1.5, 1.5],
                     ('Fe', 'Fe', 'C'): [1.5, 1.5, 1.5],
                     ('Fe', 'C', 'C'): [1.5, 1.5, 1.5],
                     ('C', 'Fe', 'Fe'): [1.5, 1.5, 1.5],
                     ('C', 'Fe', 'C'): [1.5, 1.5, 1.5],
                     ('C', 'C', 'C'): [1.5, 1.5, 1.5]}
        r_max_map = {('Fe', 'Fe'): 6.0, ('Fe', 'C'): 6.0, ('C', 'C'): 6.0,
                     ('Fe', 'Fe', 'Fe'): [5.0, 5.0, 10.0],
                     ('Fe', 'Fe', 'C'): [5.0, 5.0, 10.0],
                     ('Fe', 'C', 'C'): [5.0, 5.0, 10.0],
                     ('C', 'Fe', 'Fe'): [5.0, 5.0, 10.0],
                     ('C', 'Fe', 'C'): [5.0, 5.0, 10.0],
                     ('C', 'C', 'C'): [5.0, 5.0, 10.0]}
        resolution_map = {('Fe', 'Fe'): 12, ('Fe', 'C'): 12, ('C', 'C'): 12,
                          ('Fe', 'Fe', 'Fe'): [4, 4, 8],
                          ('Fe', 'Fe', 'C'): [4, 4, 8],
                          ('Fe', 'C', 'C'): [4, 4, 8],
                          ('C', 'Fe', 'Fe'): [4, 4, 8],
                          ('C', 'Fe', 'C'): [4, 4, 8],
                          ('C', 'C', 'C'): [4, 4, 8]}
        bspline_config = bspline.BSplineBasis(rattled_steel_chemistry,
                                              r_min_map=r_min_map,
                                              r_max_map=r_max_map,
                                              resolution_map=resolution_map,
                                              knot_strategy='linear',
                                              offset_1b=True,
                                              leading_trim=0,
                                              trailing_trim=3,)
        bspline_handler = BasisFeaturizer(bspline_config)
        eval_map = bspline_handler.evaluate_configuration(rattled_steel,
                                                          energy=0,
                                                          forces=np.zeros((3, n_atoms)))
        pkg_directory = os.path.dirname(os.path.dirname(uf3.__file__))
        data_directory = os.path.join(pkg_directory, "tests/data")
        features_file = os.path.join(data_directory, "precalculated_ref",
                                  "rattled_steel_features.json") 
        with open(features_file, 'r') as f:
            ref_features = json.load(f)
        for key in eval_map:
            # keys should be 'energy', 'fx0', 'fx1', ..., 'fx10', 'fy0', ..., 'fz10'
            assert np.allclose(eval_map[key], np.array(ref_features[key]))

def test_flatten_by_interactions():
    vector_map = {('A', 'A'): np.array([1, 1, 1]),
                  ('A', 'B'): np.array([2, 2]),
                  ('B', 'B'): np.array([3, 3, 3, 3])}
    pair_tuples = [('A', 'A'), ('A', 'B'), ('B', 'B')]
    vector = flatten_by_interactions(vector_map, pair_tuples)
    assert np.allclose(vector, [1, 1, 1, 2, 2, 3, 3, 3, 3])

