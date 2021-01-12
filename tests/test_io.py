import uf3
from uf3.data.io import *


def check_dataframe(dataframe):
    assert all([col in dataframe.columns
                for col in ['geometry', 'energy', 'size', 'fx']])
    geometry = dataframe['geometry'].iloc[0]
    assert 'energy' in geometry.info.keys()
    assert 'fx' in geometry.arrays.keys()


class TestIO:
    def test_prepare_from_lists(self):
        geometries = [ase.Atoms('Au2',
                                positions=[[0, 0, 0], [0.5, 0.3, 0.2]],
                                pbc=True,
                                cell=[[2, 0, 0], [3, 1.5, 0], [0.5, 0, 2.5]]),
                      ase.Atoms('Au3',
                                positions=[[0, 0, 0],
                                           [0.5, 0.3, 0.2],
                                           [1, 1, 1]],
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
        assert df.at[0, 'energy'] == 1.1
        assert np.allclose(df.at[0, 'fx'], (-1, 1))
        assert df.at[0, 'size'] == 2
        geometry = df.at[1, 'geometry']
        assert geometry.info['energy'] == 2.2
        assert np.allclose(geometry.arrays['fy'], (0.2, 0.3, 0.4))
        # extract energies and forces from geometries
        geometries = df['geometry'].values
        df = prepare_dataframe_from_lists(geometries, prefix='list')
        assert df.index[0] == 'list_0'
        check_dataframe(df)

    def test_parse_xyz(self):
        pkg_directory = os.path.dirname(os.path.dirname(uf3.__file__))
        data_directory = os.path.join(pkg_directory, "tests/data")
        fname = os.path.join(data_directory, "extended_xyz", "test.xyz")
        df = parse_trajectory(fname,
                              scalar_keys=['config_type'],
                              prefix='xyz')
        check_dataframe(df)
        assert df.index[0] == 'xyz_0'
        geometries = df['geometry'].values
        geometry = geometries[-1]
        assert len(geometries) == 5
        assert 'config_type' in geometry.info.keys()

    def test_parse_vasp(self):
        pkg_directory = os.path.dirname(os.path.dirname(uf3.__file__))
        data_directory = os.path.join(pkg_directory, "tests/data")
        # test static cell (ISIF < 3)
        fname = os.path.join(data_directory, "vasp_md", "vasprun.xml")
        df = parse_trajectory(fname, prefix='md')
        check_dataframe(df)
        assert df.index[0] == 'md_0'
        geometries = df['geometry'].values
        assert len(geometries) == 3
        assert np.allclose(geometries[0].cell.array,
                           geometries[-1].cell.array)

        # test changing cell (ISIF >= 3)
        fname = os.path.join(data_directory, "vasp_relax", "vasprun.xml")
        df = parse_trajectory(fname, prefix='relax')
        check_dataframe(df)
        assert df.index[0] == 'relax_0'
        geometries = df['geometry'].values
        assert len(geometries) == 3
        assert not np.allclose(geometries[0].cell.array,
                               geometries[-1].cell.array)

    def test_parse_lammps(self):
        pkg_directory = os.path.dirname(os.path.dirname(uf3.__file__))
        data_directory = os.path.join(pkg_directory, "tests/data")
        run_directory = os.path.join(data_directory, "lammps")
        df_run = parse_lammps_outputs(run_directory,
                                      prefix='lmp',
                                      dump_fname="test.lammpstrj",
                                      lammps_aliases={1: 2, 2: 10},
                                      column_subs={"PotEng": "energy"})
        check_dataframe(df_run)
        assert np.allclose(df_run['Step'],
                           [0, 1000, 2000, 3000, 0, 1000, 2000, 3000])
        assert df_run.index[0] == 'lmp_0'
        assert df_run.loc['lmp_0', 'energy'] == -477.73490


class TestDataCoordinator:
    def test_consolidate(self):
        data_handler = DataCoordinator()
        pkg_directory = os.path.dirname(os.path.dirname(uf3.__file__))
        data_directory = os.path.join(pkg_directory, "tests/data")
        run_directory = os.path.join(data_directory, "lammps")
        # LAMMPS has duplicate timesteps
        data_handler.dataframe_from_lammps_run(run_directory,
                                               prefix='lmp',
                                               dump_fname='test.lammpstrj',
                                               lammps_aliases={1: 2, 2: 10})
        relax_path = os.path.join(data_directory, "vasp_relax/vasprun.xml")
        md_path = os.path.join(data_directory, "vasp_md/vasprun.xml")
        data_handler.dataframe_from_vasprun(relax_path, prefix='vasp')
        # prefix conflict = reject
        data_handler.dataframe_from_vasprun(md_path, prefix='vasp')
        assert len(data_handler.data) == 2
        # consolidate
        df = data_handler.consolidate(remove_duplicates=True, keep='last')
        assert len(df) == 11
