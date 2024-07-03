from uf3.regression import least_squares
from pymatgen.core.structure import Structure
from pymatgen.core import Element
from uf3.data import composition
import numpy as np
import os, sys, argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Generate UF3 LAMMPS potential file")

    parser.add_argument('-a', '--author', required=True, help="Author Name Seperated by '_'")
    parser.add_argument('-u', '--units', required=True, help="LAMMPS Units")
    parser.add_argument('-m', '--model', required=True, help="UF3 Model JSON file")
    default_directory = "."
    parser.add_argument('-d', '--directory', default=default_directory,
                        help="Directory path (default: current directory)")
    default_knots_spacing_type = "nk"
    parser.add_argument('-k', '--knots_spacing_type', default=default_knots_spacing_type,
                        help="Knot spacing type, uk (uniform spacing) or \
                                nk (non-uniform spacing)\n\
                                (default: nk (non-uniform))")

    args = parser.parse_args()
    
    model = least_squares.WeightedLinearModel.from_json(args.model)

    knots_spacing_type = args.knots_spacing_type

    if (knots_spacing_type != "uk") and (knots_spacing_type != "nk"):
        raise ValueError(f"Supplied knot spacing type {knots_spacing_type} is not\n\
                 a valid choice. Only uk or nk are valid types")
    
    model_elements = set(model.bspline_config.chemical_system.element_list)
    pot_dir = args.directory

    chemical_sys = model.bspline_config.chemical_system

    uf3_lammps_pot_name = "".join(chemical_sys.element_list)+".uf3"

    write_uf3_lammps_pot_files(chemical_sys=chemical_sys, model=model,
                                knots_spacing_type = knots_spacing_type,
                                pot_dir=pot_dir,
                                uf3_lammps_pot_name = uf3_lammps_pot_name,
                                author = args.author,
                                lammps_units = args.units)
    
    lines = "pair_style\tuf3 %i %i"%(model.bspline_config.degree,len(chemical_sys.element_list))
    lines += "\n"
    lines += "pair_coeff\t* * "+pot_dir+"/"+uf3_lammps_pot_name+ " "
    lines += " ".join(chemical_sys.element_list)

    print("\n\n***Add the following line to the lammps input script***\n\n")
    print(lines)
    print("\n\nCitation meta-data has been left blank. Please enter appropriate"\
            "i citation for the generated UF3 LAMMPS potential\n\n")

def write_uf3_lammps_pot_files(chemical_sys,
                                model,
                                knots_spacing_type,
                                pot_dir,
                                uf3_lammps_pot_name,
                                author,
                                lammps_units):
    """Returns list

    Creates and writes UF3 lammps potential files. Takes UF3 composition object,
    UF3 model, knots_spacing_type, name of potential directory, name of uf3
    lammps pot file to be generateas, author and lammps units input. Will
    overwrite the files if files with the same exists
    """
    
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not os.path.exists(pot_dir):
        os.mkdir(pot_dir)
    files = {}

    for interaction in chemical_sys.interactions_map[2]:
        key = '_'.join(interaction)
        files[key] = f"#UF3 POT UNITS: {lammps_units} DATE: {current_datetime} "
        files[key] += f"AUTHOR: {author} CITATION:\n"

        files[key] += f"2B {interaction[0]} {interaction[1]}"
        files[key] += f" {model.bspline_config.leading_trim} {model.bspline_config.trailing_trim}"
        if knots_spacing_type == "uk":
            files[key] += " uk\n"
        elif knots_spacing_type == "nk":
            files[key] += " nk\n"
        else:
            raise ValueError(f"Supplied knot spacing type {knots_spacing_type}\n\
                                is not a valid choice. Only uk or nk are valid types")

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

        files[key] += "#\n"

    if 3 in model.bspline_config.interactions_map:
        for interaction in model.bspline_config.interactions_map[3]:
            key = '_'.join(interaction)
            files[key] = f"#UF3 POT UNITS: {lammps_units} DATE: {current_datetime} "
            files[key] += f"AUTHOR: {author} CITATION:\n"

            files[key] += f"3B {interaction[0]} {interaction[1]} {interaction[2]}"
            files[key] += f" {model.bspline_config.leading_trim} {model.bspline_config.trailing_trim}"
            if knots_spacing_type == "uk":
                files[key] += " uk\n"
            elif knots_spacing_type == "nk":
                files[key] += " nk\n"
            else:
                raise ValueError(f"Supplied knot spacing type {knots_spacing_type}\n\
                                    is not a valid choice. Only uk or nk are valid types")

            files[key] += str(model.bspline_config.r_max_map[interaction][2]) \
                    + " " + str(model.bspline_config.r_max_map[interaction][1]) \
                    + " " + str(model.bspline_config.r_max_map[interaction][0]) \
                    + " "

            files[key] += str(len(model.bspline_config.knots_map[interaction][2])) \
                    + " " + str(len(model.bspline_config.knots_map[interaction][1])) \
                    + " " + str(len(model.bspline_config.knots_map[interaction][0])) \
                    + "\n"

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
                    
            files[key] += "#\n"

    with open(pot_dir +"/"+ uf3_lammps_pot_name, "w") as fp:
        for k, v in files.items():
            fp.write(v)

if __name__ == "__main__":
    main()

