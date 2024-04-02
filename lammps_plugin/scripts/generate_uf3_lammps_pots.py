from uf3.regression import least_squares
from pymatgen.core.structure import Structure
from pymatgen.core import Element
from uf3.data import composition
import numpy as np
import os, sys, datetime



args = {"model_name": "", "directory": "", "author": ""}

if len(sys.argv) != 7:
    help_str = """Usage: python generate_uf3_lammps_pots.py -m <model_name> -d <directory> -a <AUTHOR>
                  Arguments:
                    -m\tModel name
                    -d\tDirectory
                    -a\tAuthor Name (Seperated by _)
                    """

    raise ValueError("Invalid number of arguments. Enter UF3 model file, name\n\
            of directory to write the UF3 potential files and author name\n"+help_str)

for i in range(1, len(sys.argv)):
    if sys.argv[i] == "-m" and i+1 < len(sys.argv):
        args["model_name"] = sys.argv[i+1]
    elif sys.argv[i] == "-d" and i+1 < len(sys.argv):
        args["directory"] = sys.argv[i+1]
    elif sys.argv[i] == "-a" and i+1 < len(sys.argv):
        args["author"] = sys.argv[i+1]

for i in args.keys():
    if args[i] == "":
        raise ValueError("Missing %s value. Specify using '-%c'"%(key, key[0]))


#struct = Structure.from_file(sys.argv[1])
model = least_squares.WeightedLinearModel.from_json(args["model_name"])

#struct_elements = set(struct.symbol_set)
model_elements = set(model.bspline_config.chemical_system.element_list)
pot_dir = args["directory"]

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
        files[key] = "#UF3 POT UNITS: metal DATE: %s AUTHOR: %s CITATION: \n"%(datetime.datetime.now(), args["author"])
        #files[key] = "#UF3 POT\n"
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
            #files[key] = "#UF3 POT\n"
            files[key] = "#UF3 POT UNITS: metal DATE: %s AUTHOR: %s CITATION: \n"%(datetime.datetime.now(), args["author"])
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


chemical_sys = model.bspline_config.chemical_system

pot_files = write_uf3_lammps_pot_files(chemical_sys=chemical_sys,model=model,pot_dir=pot_dir)
pot_files = list(pot_files)
lines = "pair_style uf3 %i %i"%(model.bspline_config.degree,len(chemical_sys.element_list))

print("\n\n***Add the following line to the lammps input script followed by the 'pair_coeff' line/s***\n\n")
print(lines)
print("\n\n")
print("CITATION metadata has been left empty")
