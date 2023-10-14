from uf3.regression import least_squares
from pymatgen.core.structure import Structure
from pymatgen.core import Element
from uf3.data import composition
import numpy as np
import os, sys

if len(sys.argv) != 3:
    #raise ValueError("Invalid number of arguments. Enter name of the structure\n\
    #        file readable by pymatgen, UF3 model file, and name \n\
    #        directory to write the UF3 potential files")
    raise ValueError("Invalid number of arguments. Enter UF3 model file and name\n\
            of directory to write the UF3 potential files")

#struct = Structure.from_file(sys.argv[1])
model = least_squares.WeightedLinearModel.from_json(sys.argv[1])

#struct_elements = set(struct.symbol_set)
model_elements = set(model.bspline_config.chemical_system.element_list)
pot_dir = sys.argv[2]

def create_element_map_for_lammps(uf3_chem_sys):
    """Returns dict

    Creates a mapping between element symbol and lammps element species,
    to be used while creating UF3 lammps file. Takes UF3 composition object as
    the input
    """
    lemap = {}
    species_count = 1
    for i in uf3_chem_sys.interactions_map[2]:
        if i[0]==i[1]:
            if i[0] not in lemap:
                lemap[i[0]] = species_count
                species_count += 1
            else:
                pass
        else:
            for j in i:
                if j not in lemap:
                    lemap[j] = species_count
                    species_count += 1
                else:
                    pass
    return lemap
"""
def write_lammps_ip_struct(struct_obj,element_map):
    """#Returns str

    #Converts a POSCAR to lammps structure format and writes 'lammps.struct' file
    #Takes pymatgen structure object as the input
"""
    struct_dict = struct_obj.as_dict()

    w_filename = 'lammps.struct'

    lx = struct_obj.lattice.a
    xy = struct_obj.lattice.b * np.cos(struct_obj.lattice.angles[2]*np.pi/180)
    xz = struct_obj.lattice.c * np.cos(struct_obj.lattice.angles[1]*np.pi/180)

    ly = np.sqrt(np.square(struct_obj.lattice.b) - np.square(xy))

    yz = ((struct_obj.lattice.b*struct_obj.lattice.c*
           np.cos(struct_obj.lattice.angles[0]*np.pi/180))-(xy*xz))/(ly)

    lz = np.sqrt(np.square(struct_obj.lattice.c)-np.square(xz)-np.square(yz))

    new_H = np.array([[lx,0,0],
         [xy,ly,0],
          [xz,yz,lz]])

    fp = open(w_filename,'w')
    fp.write("# Converted by ACH the great\n\n")
    fp.write("%i atoms\n"%struct.num_sites)
    fp.write("%i atom types\n\n"%len(struct.symbol_set))

    fp.write("0.000000%.6f   xlo xhi\n"%lx)
    fp.write("0.000000%.6f   ylo yhi\n"%ly)
    fp.write("0.000000%.6f   zlo zhi\n\n"%lz)
    fp.write("  %.6f%.6f%.6f   xy xz yz\n\n"%(xy,xz,yz))

    fp.write("Masses\n\n")

    for key in element_map:
        at_mass = Element(key)
        fp.write("  %i %.5f\n"%(element_map[key],at_mass.atomic_mass))
        fp.write("\n")

        fp.write("Atoms\n\n")
    for i in range(0,struct.num_sites):
        key = struct_dict['sites'][i]['species'][0]['element']
        temp_cord = np.matmul(struct_dict['sites'][i]['abc'],new_H)
        fp.write("   %i  %i %.6f %.6f %.6f\n"%(i+1,element_map[key],temp_cord[0],temp_cord[1],temp_cord[2]))
    fp.close()
    return w_filename
"""
def write_uf3_lammps_pot_files(chemical_sys,model,pot_dir):
    """Returns list

    Creates and writes UF3 lammps potential files. Takes UF3 composition object,
    UF3 model and name of potential directory as input. Will overwrite the files
    if files with the same exists
    """
    overwrite = True
    if not os.path.exists(pot_dir):
        os.mkdir(pot_dir)
    files = {}

    for interaction in chemical_sys.interactions_map[2]:
        key = '_'.join(interaction)
        files[key] = "#UF3 POT\n"
        if model.bspline_config.knot_strategy == 'linear':
            files[key] += "2B %i %i uk\n"%(model.bspline_config.leading_trim,model.bspline_config.trailing_trim)
        else:
            files[key] += "2B %i %i nk\n"%(model.bspline_config.leading_trim,model.bspline_config.trailing_trim)

        files[key] += str(model.bspline_config.r_max_map[interaction]) + " " + \
                str(len(model.bspline_config.knots_map[interaction]))+"\n"

        files[key] += " ".join(['{:.17g}'.format(v) for v in \
                model.bspline_config.knots_map[interaction]]) + "\n"

        files[key] += str(model.bspline_config.get_interaction_partitions()[0][interaction]) \
                + "\n"

        start_index = model.bspline_config.get_interaction_partitions()[1][interaction]
        length = model.bspline_config.get_interaction_partitions()[0][interaction]
        files[key] += " ".join(['{:.17g}'.format(v) for v in \
                model.coefficients[start_index:start_index + length]]) + "\n"

        files[key] += "#"

    if 3 in model.bspline_config.interactions_map:
        for interaction in model.bspline_config.interactions_map[3]:
            key = '_'.join(interaction)
            files[key] = "#UF3 POT\n"
            if model.bspline_config.knot_strategy == 'linear':
                files[key] += "3B %i %i uk\n"%(model.bspline_config.leading_trim,model.bspline_config.trailing_trim)
            else:
                files[key] += "3B %i %i nk\n"%(model.bspline_config.leading_trim,model.bspline_config.trailing_trim)

            files[key] += str(model.bspline_config.r_max_map[interaction][2]) \
                    + " " + str(model.bspline_config.r_max_map[interaction][1]) \
                    + " " + str(model.bspline_config.r_max_map[interaction][0]) + " "

            files[key] += str(len(model.bspline_config.knots_map[interaction][2])) \
                    + " " + str(len(model.bspline_config.knots_map[interaction][1])) + " " + str(len(model.bspline_config.knots_map[interaction][0])) + "\n"
            files[key] += " ".join(['{:.17g}'.format(v) for v in \
                    model.bspline_config.knots_map[interaction][2]]) + "\n"

            files[key] += " ".join(['{:.17g}'.format(v) for v in \
                    model.bspline_config.knots_map[interaction][1]]) + "\n"

            files[key] += " ".join(['{:.17g}'.format(v) for v in \
                    model.bspline_config.knots_map[interaction][0]]) + "\n"

            solutions = least_squares.arrange_coefficients(model.coefficients, \
                    model.bspline_config)

            decompressed = model.bspline_config.decompress_3B( \
                    solutions[(interaction[0], interaction[1],interaction[2])], \
                    (interaction[0], interaction[1],interaction[2]))

            files[key] += str(decompressed.shape[0]) + " " \
                    + str(decompressed.shape[1]) + " " \
                    + str(decompressed.shape[2]) + "\n"

            for i in range(decompressed.shape[0]):
                for j in range(decompressed.shape[1]):
                    files[key] += ' '.join(map(str, decompressed[i,j]))
                    files[key] += "\n"
                    
            files[key] += "#"

    for k, v in files.items():
        if not overwrite and os.path.exists(pot_dir + k):
            continue
        with open(pot_dir +"/"+ k, "w") as f:
            f.write(v)
    return files.keys()

"""
if len(model_elements.intersection(struct_elements))==len(struct_elements):
    chemical_sys = composition.ChemicalSystem(element_list=list(struct_elements),\
            degree=model.bspline_config.degree)
    element_map = create_element_map_for_lammps(chemical_sys)
    print(element_map)
    write_lammps_ip_struct(struct_obj=struct,element_map=element_map)
    pot_files = write_uf3_lammps_pot_files(chemical_sys=chemical_sys,model=model,pot_dir=pot_dir)
    pot_files = list(pot_files)
    lines = "pair_style uf3 %i %i\n"%(model.bspline_config.degree,len(struct_elements))

    for interaction in chemical_sys.interactions_map[2]:
        lines += "pair_coeff %i %i %s/%s\n"%(element_map[interaction[0]],
                element_map[interaction[1]],pot_dir,'_'.join(interaction))

    if 3 in model.bspline_config.interactions_map:
        for interaction in model.bspline_config.interactions_map[3]:
            lines += "pair_coeff %i %i %i %s/%s\n"%(element_map[interaction[0]],
                    element_map[interaction[1]],element_map[interaction[2]],
                    pot_dir,'_'.join(interaction))

    print("***Add the following lines to the lammps input script***")
    print(lines)
else:
    raise RuntimeError("Elements in the provided structure file does not match the elements in ")
"""

chemical_sys = model.bspline_config.chemical_system

pot_files = write_uf3_lammps_pot_files(chemical_sys=chemical_sys,model=model,pot_dir=pot_dir)
pot_files = list(pot_files)
lines = "pair_style uf3 %i %i"%(model.bspline_config.degree,len(chemical_sys.element_list))

"""
for interaction in chemical_sys.interactions_map[2]:
    lines += " %s/%s"%(pot_dir,'_'.join(interaction))

if 3 in model.bspline_config.interactions_map:
    for interaction in model.bspline_config.interactions_map[3]:
        lines += " %s/%s"%(pot_dir,'_'.join(interaction))
"""
print("\n\n***Add the following line to the lammps input script followed by the 'pair_coeff' line/s***\n\n")
print(lines)
print("\n\n")
