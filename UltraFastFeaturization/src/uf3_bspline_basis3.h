#include <cmath>
#include <vector>

std::vector<double> get_constants(double *knots)
{

  std::vector<double> constants(16);

  constants[0] = (-pow(knots[0], 3) /
       (-pow(knots[0], 3) + pow(knots[0], 2) * knots[1] + pow(knots[0], 2) * knots[2] +
        pow(knots[0], 2) * knots[3] - knots[0] * knots[1] * knots[2] -
        knots[0] * knots[1] * knots[3] - knots[0] * knots[2] * knots[3] +
        knots[1] * knots[2] * knots[3]));
  constants[1] = (3 * pow(knots[0], 2) /
       (-pow(knots[0], 3) + pow(knots[0], 2) * knots[1] + pow(knots[0], 2) * knots[2] +
        pow(knots[0], 2) * knots[3] - knots[0] * knots[1] * knots[2] -
        knots[0] * knots[1] * knots[3] - knots[0] * knots[2] * knots[3] +
        knots[1] * knots[2] * knots[3]));
  constants[2] = (-3 * knots[0] /
       (-pow(knots[0], 3) + pow(knots[0], 2) * knots[1] + pow(knots[0], 2) * knots[2] +
        pow(knots[0], 2) * knots[3] - knots[0] * knots[1] * knots[2] -
        knots[0] * knots[1] * knots[3] - knots[0] * knots[2] * knots[3] +
        knots[1] * knots[2] * knots[3]));
  constants[3] = (1 /
       (-pow(knots[0], 3) + pow(knots[0], 2) * knots[1] + pow(knots[0], 2) * knots[2] +
        pow(knots[0], 2) * knots[3] - knots[0] * knots[1] * knots[2] -
        knots[0] * knots[1] * knots[3] - knots[0] * knots[2] * knots[3] +
        knots[1] * knots[2] * knots[3]));
  constants[4] = (pow(knots[1], 2) * knots[4] /
           (-pow(knots[1], 3) + pow(knots[1], 2) * knots[2] + pow(knots[1], 2) * knots[3] +
            pow(knots[1], 2) * knots[4] - knots[1] * knots[2] * knots[3] -
            knots[1] * knots[2] * knots[4] - knots[1] * knots[3] * knots[4] +
            knots[2] * knots[3] * knots[4]) +
       pow(knots[0], 2) * knots[2] /
           (-pow(knots[0], 2) * knots[1] + pow(knots[0], 2) * knots[2] +
            knots[0] * knots[1] * knots[2] + knots[0] * knots[1] * knots[3] -
            knots[0] * pow(knots[2], 2) - knots[0] * knots[2] * knots[3] -
            knots[1] * knots[2] * knots[3] + pow(knots[2], 2) * knots[3]) +
       knots[0] * knots[1] * knots[3] /
           (-knots[0] * pow(knots[1], 2) + knots[0] * knots[1] * knots[2] +
            knots[0] * knots[1] * knots[3] - knots[0] * knots[2] * knots[3] +
            pow(knots[1], 2) * knots[3] - knots[1] * knots[2] * knots[3] -
            knots[1] * pow(knots[3], 2) + knots[2] * pow(knots[3], 2)));
  constants[5] = (-pow(knots[1], 2) /
           (-pow(knots[1], 3) + pow(knots[1], 2) * knots[2] + pow(knots[1], 2) * knots[3] +
            pow(knots[1], 2) * knots[4] - knots[1] * knots[2] * knots[3] -
            knots[1] * knots[2] * knots[4] - knots[1] * knots[3] * knots[4] +
            knots[2] * knots[3] * knots[4]) -
       2 * knots[1] * knots[4] /
           (-pow(knots[1], 3) + pow(knots[1], 2) * knots[2] + pow(knots[1], 2) * knots[3] +
            pow(knots[1], 2) * knots[4] - knots[1] * knots[2] * knots[3] -
            knots[1] * knots[2] * knots[4] - knots[1] * knots[3] * knots[4] +
            knots[2] * knots[3] * knots[4]) -
       pow(knots[0], 2) /
           (-pow(knots[0], 2) * knots[1] + pow(knots[0], 2) * knots[2] +
            knots[0] * knots[1] * knots[2] + knots[0] * knots[1] * knots[3] -
            knots[0] * pow(knots[2], 2) - knots[0] * knots[2] * knots[3] -
            knots[1] * knots[2] * knots[3] + pow(knots[2], 2) * knots[3]) -
       2 * knots[0] * knots[2] /
           (-pow(knots[0], 2) * knots[1] + pow(knots[0], 2) * knots[2] +
            knots[0] * knots[1] * knots[2] + knots[0] * knots[1] * knots[3] -
            knots[0] * pow(knots[2], 2) - knots[0] * knots[2] * knots[3] -
            knots[1] * knots[2] * knots[3] + pow(knots[2], 2) * knots[3]) -
       knots[0] * knots[1] /
           (-knots[0] * pow(knots[1], 2) + knots[0] * knots[1] * knots[2] +
            knots[0] * knots[1] * knots[3] - knots[0] * knots[2] * knots[3] +
            pow(knots[1], 2) * knots[3] - knots[1] * knots[2] * knots[3] -
            knots[1] * pow(knots[3], 2) + knots[2] * pow(knots[3], 2)) -
       knots[0] * knots[3] /
           (-knots[0] * pow(knots[1], 2) + knots[0] * knots[1] * knots[2] +
            knots[0] * knots[1] * knots[3] - knots[0] * knots[2] * knots[3] +
            pow(knots[1], 2) * knots[3] - knots[1] * knots[2] * knots[3] -
            knots[1] * pow(knots[3], 2) + knots[2] * pow(knots[3], 2)) -
       knots[1] * knots[3] /
           (-knots[0] * pow(knots[1], 2) + knots[0] * knots[1] * knots[2] +
            knots[0] * knots[1] * knots[3] - knots[0] * knots[2] * knots[3] +
            pow(knots[1], 2) * knots[3] - knots[1] * knots[2] * knots[3] -
            knots[1] * pow(knots[3], 2) + knots[2] * pow(knots[3], 2)));
  constants[6] = (2 * knots[1] /
           (-pow(knots[1], 3) + pow(knots[1], 2) * knots[2] + pow(knots[1], 2) * knots[3] +
            pow(knots[1], 2) * knots[4] - knots[1] * knots[2] * knots[3] -
            knots[1] * knots[2] * knots[4] - knots[1] * knots[3] * knots[4] +
            knots[2] * knots[3] * knots[4]) +
       knots[4] /
           (-pow(knots[1], 3) + pow(knots[1], 2) * knots[2] + pow(knots[1], 2) * knots[3] +
            pow(knots[1], 2) * knots[4] - knots[1] * knots[2] * knots[3] -
            knots[1] * knots[2] * knots[4] - knots[1] * knots[3] * knots[4] +
            knots[2] * knots[3] * knots[4]) +
       2 * knots[0] /
           (-pow(knots[0], 2) * knots[1] + pow(knots[0], 2) * knots[2] +
            knots[0] * knots[1] * knots[2] + knots[0] * knots[1] * knots[3] -
            knots[0] * pow(knots[2], 2) - knots[0] * knots[2] * knots[3] -
            knots[1] * knots[2] * knots[3] + pow(knots[2], 2) * knots[3]) +
       knots[2] /
           (-pow(knots[0], 2) * knots[1] + pow(knots[0], 2) * knots[2] +
            knots[0] * knots[1] * knots[2] + knots[0] * knots[1] * knots[3] -
            knots[0] * pow(knots[2], 2) - knots[0] * knots[2] * knots[3] -
            knots[1] * knots[2] * knots[3] + pow(knots[2], 2) * knots[3]) +
       knots[0] /
           (-knots[0] * pow(knots[1], 2) + knots[0] * knots[1] * knots[2] +
            knots[0] * knots[1] * knots[3] - knots[0] * knots[2] * knots[3] +
            pow(knots[1], 2) * knots[3] - knots[1] * knots[2] * knots[3] -
            knots[1] * pow(knots[3], 2) + knots[2] * pow(knots[3], 2)) +
       knots[1] /
           (-knots[0] * pow(knots[1], 2) + knots[0] * knots[1] * knots[2] +
            knots[0] * knots[1] * knots[3] - knots[0] * knots[2] * knots[3] +
            pow(knots[1], 2) * knots[3] - knots[1] * knots[2] * knots[3] -
            knots[1] * pow(knots[3], 2) + knots[2] * pow(knots[3], 2)) +
       knots[3] /
           (-knots[0] * pow(knots[1], 2) + knots[0] * knots[1] * knots[2] +
            knots[0] * knots[1] * knots[3] - knots[0] * knots[2] * knots[3] +
            pow(knots[1], 2) * knots[3] - knots[1] * knots[2] * knots[3] -
            knots[1] * pow(knots[3], 2) + knots[2] * pow(knots[3], 2)));
  constants[7] = (-1 /
           (-pow(knots[1], 3) + pow(knots[1], 2) * knots[2] + pow(knots[1], 2) * knots[3] +
            pow(knots[1], 2) * knots[4] - knots[1] * knots[2] * knots[3] -
            knots[1] * knots[2] * knots[4] - knots[1] * knots[3] * knots[4] +
            knots[2] * knots[3] * knots[4]) -
       1 /
           (-pow(knots[0], 2) * knots[1] + pow(knots[0], 2) * knots[2] +
            knots[0] * knots[1] * knots[2] + knots[0] * knots[1] * knots[3] -
            knots[0] * pow(knots[2], 2) - knots[0] * knots[2] * knots[3] -
            knots[1] * knots[2] * knots[3] + pow(knots[2], 2) * knots[3]) -
       1 /
           (-knots[0] * pow(knots[1], 2) + knots[0] * knots[1] * knots[2] +
            knots[0] * knots[1] * knots[3] - knots[0] * knots[2] * knots[3] +
            pow(knots[1], 2) * knots[3] - knots[1] * knots[2] * knots[3] -
            knots[1] * pow(knots[3], 2) + knots[2] * pow(knots[3], 2)));
  constants[8] = (-knots[0] * pow(knots[3], 2) /
           (-knots[0] * knots[1] * knots[2] + knots[0] * knots[1] * knots[3] +
            knots[0] * knots[2] * knots[3] - knots[0] * pow(knots[3], 2) +
            knots[1] * knots[2] * knots[3] - knots[1] * pow(knots[3], 2) -
            knots[2] * pow(knots[3], 2) + pow(knots[3], 3)) -
       knots[1] * knots[3] * knots[4] /
           (-pow(knots[1], 2) * knots[2] + pow(knots[1], 2) * knots[3] +
            knots[1] * knots[2] * knots[3] + knots[1] * knots[2] * knots[4] -
            knots[1] * pow(knots[3], 2) - knots[1] * knots[3] * knots[4] -
            knots[2] * knots[3] * knots[4] + pow(knots[3], 2) * knots[4]) -
       knots[2] * pow(knots[4], 2) /
           (-knots[1] * pow(knots[2], 2) + knots[1] * knots[2] * knots[3] +
            knots[1] * knots[2] * knots[4] - knots[1] * knots[3] * knots[4] +
            pow(knots[2], 2) * knots[4] - knots[2] * knots[3] * knots[4] -
            knots[2] * pow(knots[4], 2) + knots[3] * pow(knots[4], 2)));
  constants[9] = (2 * knots[0] * knots[3] /
           (-knots[0] * knots[1] * knots[2] + knots[0] * knots[1] * knots[3] +
            knots[0] * knots[2] * knots[3] - knots[0] * pow(knots[3], 2) +
            knots[1] * knots[2] * knots[3] - knots[1] * pow(knots[3], 2) -
            knots[2] * pow(knots[3], 2) + pow(knots[3], 3)) +
       pow(knots[3], 2) /
           (-knots[0] * knots[1] * knots[2] + knots[0] * knots[1] * knots[3] +
            knots[0] * knots[2] * knots[3] - knots[0] * pow(knots[3], 2) +
            knots[1] * knots[2] * knots[3] - knots[1] * pow(knots[3], 2) -
            knots[2] * pow(knots[3], 2) + pow(knots[3], 3)) +
       knots[1] * knots[3] /
           (-pow(knots[1], 2) * knots[2] + pow(knots[1], 2) * knots[3] +
            knots[1] * knots[2] * knots[3] + knots[1] * knots[2] * knots[4] -
            knots[1] * pow(knots[3], 2) - knots[1] * knots[3] * knots[4] -
            knots[2] * knots[3] * knots[4] + pow(knots[3], 2) * knots[4]) +
       knots[1] * knots[4] /
           (-pow(knots[1], 2) * knots[2] + pow(knots[1], 2) * knots[3] +
            knots[1] * knots[2] * knots[3] + knots[1] * knots[2] * knots[4] -
            knots[1] * pow(knots[3], 2) - knots[1] * knots[3] * knots[4] -
            knots[2] * knots[3] * knots[4] + pow(knots[3], 2) * knots[4]) +
       knots[3] * knots[4] /
           (-pow(knots[1], 2) * knots[2] + pow(knots[1], 2) * knots[3] +
            knots[1] * knots[2] * knots[3] + knots[1] * knots[2] * knots[4] -
            knots[1] * pow(knots[3], 2) - knots[1] * knots[3] * knots[4] -
            knots[2] * knots[3] * knots[4] + pow(knots[3], 2) * knots[4]) +
       2 * knots[2] * knots[4] /
           (-knots[1] * pow(knots[2], 2) + knots[1] * knots[2] * knots[3] +
            knots[1] * knots[2] * knots[4] - knots[1] * knots[3] * knots[4] +
            pow(knots[2], 2) * knots[4] - knots[2] * knots[3] * knots[4] -
            knots[2] * pow(knots[4], 2) + knots[3] * pow(knots[4], 2)) +
       pow(knots[4], 2) /
           (-knots[1] * pow(knots[2], 2) + knots[1] * knots[2] * knots[3] +
            knots[1] * knots[2] * knots[4] - knots[1] * knots[3] * knots[4] +
            pow(knots[2], 2) * knots[4] - knots[2] * knots[3] * knots[4] -
            knots[2] * pow(knots[4], 2) + knots[3] * pow(knots[4], 2)));
  constants[10] = (-knots[0] /
           (-knots[0] * knots[1] * knots[2] + knots[0] * knots[1] * knots[3] +
            knots[0] * knots[2] * knots[3] - knots[0] * pow(knots[3], 2) +
            knots[1] * knots[2] * knots[3] - knots[1] * pow(knots[3], 2) -
            knots[2] * pow(knots[3], 2) + pow(knots[3], 3)) -
       2 * knots[3] /
           (-knots[0] * knots[1] * knots[2] + knots[0] * knots[1] * knots[3] +
            knots[0] * knots[2] * knots[3] - knots[0] * pow(knots[3], 2) +
            knots[1] * knots[2] * knots[3] - knots[1] * pow(knots[3], 2) -
            knots[2] * pow(knots[3], 2) + pow(knots[3], 3)) -
       knots[1] /
           (-pow(knots[1], 2) * knots[2] + pow(knots[1], 2) * knots[3] +
            knots[1] * knots[2] * knots[3] + knots[1] * knots[2] * knots[4] -
            knots[1] * pow(knots[3], 2) - knots[1] * knots[3] * knots[4] -
            knots[2] * knots[3] * knots[4] + pow(knots[3], 2) * knots[4]) -
       knots[3] /
           (-pow(knots[1], 2) * knots[2] + pow(knots[1], 2) * knots[3] +
            knots[1] * knots[2] * knots[3] + knots[1] * knots[2] * knots[4] -
            knots[1] * pow(knots[3], 2) - knots[1] * knots[3] * knots[4] -
            knots[2] * knots[3] * knots[4] + pow(knots[3], 2) * knots[4]) -
       knots[4] /
           (-pow(knots[1], 2) * knots[2] + pow(knots[1], 2) * knots[3] +
            knots[1] * knots[2] * knots[3] + knots[1] * knots[2] * knots[4] -
            knots[1] * pow(knots[3], 2) - knots[1] * knots[3] * knots[4] -
            knots[2] * knots[3] * knots[4] + pow(knots[3], 2) * knots[4]) -
       knots[2] /
           (-knots[1] * pow(knots[2], 2) + knots[1] * knots[2] * knots[3] +
            knots[1] * knots[2] * knots[4] - knots[1] * knots[3] * knots[4] +
            pow(knots[2], 2) * knots[4] - knots[2] * knots[3] * knots[4] -
            knots[2] * pow(knots[4], 2) + knots[3] * pow(knots[4], 2)) -
       2 * knots[4] /
           (-knots[1] * pow(knots[2], 2) + knots[1] * knots[2] * knots[3] +
            knots[1] * knots[2] * knots[4] - knots[1] * knots[3] * knots[4] +
            pow(knots[2], 2) * knots[4] - knots[2] * knots[3] * knots[4] -
            knots[2] * pow(knots[4], 2) + knots[3] * pow(knots[4], 2)));
  constants[11] = (1 /
           (-knots[0] * knots[1] * knots[2] + knots[0] * knots[1] * knots[3] +
            knots[0] * knots[2] * knots[3] - knots[0] * pow(knots[3], 2) +
            knots[1] * knots[2] * knots[3] - knots[1] * pow(knots[3], 2) -
            knots[2] * pow(knots[3], 2) + pow(knots[3], 3)) +
       1 /
           (-pow(knots[1], 2) * knots[2] + pow(knots[1], 2) * knots[3] +
            knots[1] * knots[2] * knots[3] + knots[1] * knots[2] * knots[4] -
            knots[1] * pow(knots[3], 2) - knots[1] * knots[3] * knots[4] -
            knots[2] * knots[3] * knots[4] + pow(knots[3], 2) * knots[4]) +
       1 /
           (-knots[1] * pow(knots[2], 2) + knots[1] * knots[2] * knots[3] +
            knots[1] * knots[2] * knots[4] - knots[1] * knots[3] * knots[4] +
            pow(knots[2], 2) * knots[4] - knots[2] * knots[3] * knots[4] -
            knots[2] * pow(knots[4], 2) + knots[3] * pow(knots[4], 2)));
  constants[12] = (pow(knots[4], 3) /
       (-knots[1] * knots[2] * knots[3] + knots[1] * knots[2] * knots[4] +
        knots[1] * knots[3] * knots[4] - knots[1] * pow(knots[4], 2) +
        knots[2] * knots[3] * knots[4] - knots[2] * pow(knots[4], 2) - knots[3] * pow(knots[4], 2) +
        pow(knots[4], 3)));
  constants[13] = (-3 * pow(knots[4], 2) /
       (-knots[1] * knots[2] * knots[3] + knots[1] * knots[2] * knots[4] +
        knots[1] * knots[3] * knots[4] - knots[1] * pow(knots[4], 2) +
        knots[2] * knots[3] * knots[4] - knots[2] * pow(knots[4], 2) - knots[3] * pow(knots[4], 2) +
        pow(knots[4], 3)));
  constants[14] = (3 * knots[4] /
       (-knots[1] * knots[2] * knots[3] + knots[1] * knots[2] * knots[4] +
        knots[1] * knots[3] * knots[4] - knots[1] * pow(knots[4], 2) +
        knots[2] * knots[3] * knots[4] - knots[2] * pow(knots[4], 2) - knots[3] * pow(knots[4], 2) +
        pow(knots[4], 3)));
  constants[15] = (-1 /
       (-knots[1] * knots[2] * knots[3] + knots[1] * knots[2] * knots[4] +
        knots[1] * knots[3] * knots[4] - knots[1] * pow(knots[4], 2) +
        knots[2] * knots[3] * knots[4] - knots[2] * pow(knots[4], 2) - knots[3] * pow(knots[4], 2) +
        pow(knots[4], 3)));

  return constants;
}
