import ufpotential
from ufpotential.data.io import *

class TestIO:
    def test_prepare_from_lists(self):
        geometries = [Atoms('Au2',
                            positions=[[0, 0, 0], [0.5, 0.3, 0.2]],
                            pbc=True,
                            cell=[[2, 0, 0], [3, 1.5, 0], [0.5, 0, 2.5]]),
                      Atoms('Au3',
                            positions=[[0, 0, 0], [0.5, 0.3, 0.2], [1, 1, 1]],
                            pbc=True,
                            cell=[[2, 0, 0], [3, 1.5, 0], [0.5, 0, 2.5]])]
        energies = [1.1, 2.2]
        forces = [[[-1, -0.6, -0.4],
                   [1, 0.6, 0.4]],
                  [[0.1, 0.2, 0.3],
                   [0.2, 0.3, 0.4],
                   [0.3, 0.4, 0.5]]]
        # providing energies and forces as lists
        df = prepare_dataframe_from_lists(geometries,
                                          energies=energies,
                                          forces=forces)
        assert 'energy' in df.columns
        assert 'fx' in df.columns
        geometry = df['geometry'].iloc[0]
        assert 'energy' in geometry.info.keys()
        assert 'fx' in geometry.arrays.keys()
        # extract energies and forces from geometries
        geometries = df['geometry'].values
        df = prepare_dataframe_from_lists(geometries)
        assert 'energy' in df.columns
        assert 'fx' in df.columns
        geometry = df['geometry'].iloc[0]
        assert 'energy' in geometry.info.keys()
        assert 'fx' in geometry.arrays.keys()

    def test_parse_xyz(self):
        data_directory = os.path.join(os.path.dirname(ufpotential.__file__),
                                      "tests/data")
        fname = os.path.join(data_directory, "extended_xyz", "test.xyz")
        df = parse_trajectory(fname, scalar_keys=['config_type'])
        assert 'energy' in df.columns
        assert 'config_type' in df.columns
        assert 'fx' in df.columns
        geometries = df['geometry'].values
        geometry = geometries[-1]
        assert len(geometries) == 5
        assert 'config_type' in geometry.info.keys()
        assert 'energy' in geometry.info.keys()
        assert 'fx' in geometry.arrays.keys()


    def test_parse_vasp(self):
        data_directory = os.path.join(os.path.dirname(ufpotential.__file__),
                                      "tests/data")
        # test static cell (ISIF < 3)
        fname = os.path.join(data_directory, "vasp_md", "vasprun.xml")
        df = parse_trajectory(fname, prefix='md')
        assert 'energy' in df.columns
        assert 'fx' in df.columns
        assert df.index[0] == 'md_0'
        geometries = df['geometry'].values
        assert len(geometries) == 3
        assert np.allclose(geometries[0].cell.array,
                           geometries[-1].cell.array)
        geometry = geometries[-1]
        assert 'energy' in geometry.info.keys()
        assert 'fx' in geometry.arrays.keys()

        # test changing cell (ISIF >= 3)
        fname = os.path.join(data_directory, "vasp_relax", "vasprun.xml")
        df = parse_trajectory(fname, prefix='relax')
        assert 'energy' in df.columns
        assert 'fx' in df.columns
        assert df.index[0] == 'relax_0'
        geometries = df['geometry'].values
        assert len(geometries) == 3
        assert not np.allclose(geometries[0].cell.array,
                               geometries[-1].cell.array)
        geometry = geometries[-1]
        assert 'energy' in geometry.info.keys()
        assert 'fx' in geometry.arrays.keys()

    def test_parse_lammps(self):
        run_directory = os.path.join(os.path.dirname(ufpotential.__file__),
                                     "tests/data/lammps")
        df_run = parse_lammps_outputs(run_directory,
                                      dump_fname="test.lammpstrj",
                                      element_aliases={1: 2, 2: 10})
        assert len(df_run) == 8
        assert np.allclose(df_run['Step'],
                           [0, 1000, 2000, 3000, 0, 1000, 2000, 3000])
        geometry = df_run['geometry'].iloc[0]
        assert 'energy' in geometry.info.keys()
        assert 'fx' in geometry.arrays.keys()
        print(df_run['energy'])
        print(df_run.loc[0, 'energy'])
        assert np.allclose(df_run.loc[0, 'energy'], (-477.73490, -418.58648))


