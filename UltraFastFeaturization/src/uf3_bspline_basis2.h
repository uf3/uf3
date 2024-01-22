#include <cmath>
#include <vector>

std::vector<double> get_dnconstants(double *knots, double coefficient)
{
  std::vector<double> constants(9);

  constants[0] = coefficient *
      (pow(knots[0], 2) /
       (pow(knots[0], 2) - knots[0] * knots[1] - knots[0] * knots[2] + knots[1] * knots[2]));
  constants[1] = coefficient *
      (-2 * knots[0] /
       (pow(knots[0], 2) - knots[0] * knots[1] - knots[0] * knots[2] + knots[1] * knots[2]));
  constants[2] = coefficient *
      (1 / (pow(knots[0], 2) - knots[0] * knots[1] - knots[0] * knots[2] + knots[1] * knots[2]));
  constants[3] = coefficient *
      (-knots[1] * knots[3] /
           (pow(knots[1], 2) - knots[1] * knots[2] - knots[1] * knots[3] + knots[2] * knots[3]) -
       knots[0] * knots[2] /
           (knots[0] * knots[1] - knots[0] * knots[2] - knots[1] * knots[2] + pow(knots[2], 2)));
  constants[4] = coefficient *
      (knots[1] /
           (pow(knots[1], 2) - knots[1] * knots[2] - knots[1] * knots[3] + knots[2] * knots[3]) +
       knots[3] /
           (pow(knots[1], 2) - knots[1] * knots[2] - knots[1] * knots[3] + knots[2] * knots[3]) +
       knots[0] /
           (knots[0] * knots[1] - knots[0] * knots[2] - knots[1] * knots[2] + pow(knots[2], 2)) +
       knots[2] /
           (knots[0] * knots[1] - knots[0] * knots[2] - knots[1] * knots[2] + pow(knots[2], 2)));
  constants[5] = coefficient *
      (-1 / (pow(knots[1], 2) - knots[1] * knots[2] - knots[1] * knots[3] + knots[2] * knots[3]) -
       1 / (knots[0] * knots[1] - knots[0] * knots[2] - knots[1] * knots[2] + pow(knots[2], 2)));
  constants[6] = coefficient *
      (pow(knots[3], 2) /
       (knots[1] * knots[2] - knots[1] * knots[3] - knots[2] * knots[3] + pow(knots[3], 2)));
  constants[7] = coefficient *
      (-2 * knots[3] /
       (knots[1] * knots[2] - knots[1] * knots[3] - knots[2] * knots[3] + pow(knots[3], 2)));
  constants[8] = coefficient *
      (1 / (knots[1] * knots[2] - knots[1] * knots[3] - knots[2] * knots[3] + pow(knots[3], 2)));

  return constants;
}
