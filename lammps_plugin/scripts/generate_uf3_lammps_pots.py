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
pot_file = sys.argv[2]

def create_element_map_for_lammps(uf3_chem_sys):
    """Returns dict

    Creates a mapping between element symbol and lammps element species,
    to be used while creating UF3 lammps file. Takes UF3 composition object as
    the input
    """
    elements = []
    result = {}

    for element_pair in uf3_chem_sys.interactions_map[2]:
        i, j = element_pair
        if i not in elements: 
            elements.append(i)
        if j not in elements: 
            elements.append(j)

    elements = list(elements)
    elements = sorted(elements)
    for i, e in enumerate(elements):
        result[i+1] = e

    return result

def write_uf3_lammps_pot_file(chemical_sys, model, pot_file, units='metal') -> None:
    """Returns list

    Creates and writes UF3 lammps potential files. Takes UF3 composition object,
    UF3 model and name of potential directory as input. Will overwrite the files
    if files with the same exists
    """
    result = ""

    for interaction in chemical_sys.interactions_map[2]:
        result += "#UF3 POT UNITS: %s \n" % units
        if model.bspline_config.knot_strategy == 'linear':
            result += "2B %s %s %i %i uk\n" % (interaction[0], interaction[1], model.bspline_config.leading_trim,model.bspline_config.trailing_trim)
        else:
            result += "2B %s %s %i %i nk\n" % (interaction[0], interaction[1], model.bspline_config.leading_trim,model.bspline_config.trailing_trim)

        result += str(model.bspline_config.r_max_map[interaction]) + " " + \
                str(len(model.bspline_config.knots_map[interaction]))+"\n"

        result += " ".join(['{:.17g}'.format(v) for v in \
                model.bspline_config.knots_map[interaction]]) + "\n"

        result += str(model.bspline_config.get_interaction_partitions()[0][interaction]) \
                + "\n"

        start_index = model.bspline_config.get_interaction_partitions()[1][interaction]
        length = model.bspline_config.get_interaction_partitions()[0][interaction]
        result += " ".join(['{:.17g}'.format(v) for v in \
                model.coefficients[start_index:start_index + length]]) + "\n"

        result += "# \n"

    if 3 in model.bspline_config.interactions_map:
        for interaction in model.bspline_config.interactions_map[3]:
            result += "#UF3 POT UNITS: %s \n" % units
            if model.bspline_config.knot_strategy == 'linear':
                result += "3B %s %s %s %i %i uk\n" % (interaction[0], interaction[1], interaction[2], model.bspline_config.leading_trim, model.bspline_config.trailing_trim)
            else:
                result += "3B %s %s %s %i %i nk\n" % (interaction[0], interaction[1], interaction[2], model.bspline_config.leading_trim, model.bspline_config.trailing_trim)

            result += str(model.bspline_config.r_max_map[interaction][2]) \
                    + " " + str(model.bspline_config.r_max_map[interaction][1]) \
                    + " " + str(model.bspline_config.r_max_map[interaction][0]) + " "

            result += str(len(model.bspline_config.knots_map[interaction][2])) \
                    + " " + str(len(model.bspline_config.knots_map[interaction][1])) + " " + str(len(model.bspline_config.knots_map[interaction][0])) + "\n"
            result += " ".join(['{:.17g}'.format(v) for v in \
                    model.bspline_config.knots_map[interaction][2]]) + "\n"

            result += " ".join(['{:.17g}'.format(v) for v in \
                    model.bspline_config.knots_map[interaction][1]]) + "\n"

            result += " ".join(['{:.17g}'.format(v) for v in \
                    model.bspline_config.knots_map[interaction][0]]) + "\n"

            solutions = least_squares.arrange_coefficients(model.coefficients, \
                    model.bspline_config)

            decompressed = model.bspline_config.decompress_3B( \
                    solutions[(interaction[0], interaction[1],interaction[2])], \
                    (interaction[0], interaction[1],interaction[2]))

            result += str(decompressed.shape[0]) + " " \
                    + str(decompressed.shape[1]) + " " \
                    + str(decompressed.shape[2]) + "\n"

            for i in range(decompressed.shape[0]):
                for j in range(decompressed.shape[1]):
                    result += ' '.join(map(str, decompressed[i,j]))
                    result += "\n"
                    
            result += "# \n"

    with open(pot_file, 'w') as f:
        f.write(result)

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

write_uf3_lammps_pot_file(chemical_sys=chemical_sys, model=model, pot_file=pot_file)
# pot_files = list(pot_files)
lines = "pair_style uf3 %i \n" % model.bspline_config.degree

# pair_coeff line may have any ordering of the species depending on input structure
# alphabetical is the ASE default I think
lines += "pair_coeff * * %s" % pot_file

element_map = create_element_map_for_lammps(chemical_sys)
elements_in_order = []
for i in range(1, len(element_map)+1):
    elements_in_order.append(element_map[i])
lines += " " + ' '.join(elements_in_order) + '\n'

"""
for interaction in chemical_sys.interactions_map[2]:
    lines += " %s/%s"%(pot_dir,'_'.join(interaction))

if 3 in model.bspline_config.interactions_map:
    for interaction in model.bspline_config.interactions_map[3]:
        lines += " %s/%s"%(pot_dir,'_'.join(interaction))
"""

print("\n\n***Add the following line to the lammps input script followed by the 'pair_coeff' line/s***\n\n")
print(lines)
print("\n")

print('Please double check the order of the elements for pair_coeff for your use case.')
