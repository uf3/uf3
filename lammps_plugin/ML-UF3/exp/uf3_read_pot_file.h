#include "pointers.h"

#include <vector>

#ifndef UF3_READ_POT_FILE_H
#define UF3_READ_POT_FILE_H

namespace LAMMPS_NS {

class uf3_read_pot_file
{
public:
  uf3_read_pot_file();
  uf3_read_pot_file(char *potf_name);
  ~uf3_read_pot_file();
  void read_file();
};
}
#endif
