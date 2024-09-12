#include <iostream>
#include <iomanip>
#include <fstream>
#include <omp.h>
#include <cmath>
#include <vector>

double func(double x)
{
  return sin(x);
}

double calc(double x0, double x1, double dx, uint32_t num_threads)
{
  std::vector<double> sum(num_threads);
  std::vector<double> err(num_threads);
  uint32_t steps = uint32_t((x1 - x0) / dx);

  #pragma omp parallel for num_threads(num_threads)
  for(uint32_t i = 0 ; i < steps; ++i)
  {
    double x     = x0 + i * dx;
    double input = (func(x) + func(x + dx)) / 2 * dx;
    double t     = sum[omp_get_thread_num()] + input;

    err[omp_get_thread_num()] += sum[omp_get_thread_num()] - t + input;
    sum[omp_get_thread_num()] = t;
  }

  double res = 0;
  for(uint32_t i = 0; i < num_threads; i++)
  {
    res += sum[i] + err[i];
  }
  res += (func(steps * dx + x0) + func(x1)) / 2 * dx;
  return res;
}

int main(int argc, char** argv)
{
  // Check arguments
  if (argc != 3)
  {
    std::cout << "[Error] Usage <inputfile> <output file>\n";
    return 1;
  }

  // Prepare input file
  std::ifstream input(argv[1]);
  if (!input.is_open())
  {
    std::cout << "[Error] Can't open " << argv[1] << " for write\n";
    return 1;
  }

  // Prepare output file
  std::ofstream output(argv[2]);
  if (!output.is_open())
  {
    std::cout << "[Error] Can't open " << argv[2] << " for read\n";
    input.close();
    return 1;
  }

  // Read arguments from input
  double x0 = 0.0, x1 =0.0, dx = 0.0;
  uint32_t num_threads = 0;
  input >> x0 >> x1 >> dx >> num_threads;

  // Calculation
  double res = calc(x0, x1, dx, num_threads);

  // Write result
  output << std::setprecision(13) << res << std::endl;
  // Prepare to exit
  output.close();
  input.close();
  return 0;
}
