#include <iostream>
#include <iomanip>
#include <fstream>
#include <omp.h>

double calc(uint32_t x_last, uint32_t num_threads)
{
  double* sum = (double*)calloc(num_threads, sizeof(double));
  double* err = (double*)calloc(num_threads, sizeof(double));

  #pragma omp parallel for num_threads(num_threads)
  for(uint32_t i = 1; i <= x_last; i++)
  {
  	double t = sum[omp_get_thread_num()] + 1 / (double) i;
    err[omp_get_thread_num()] += sum[omp_get_thread_num()] - t + 1 / (double) i;
    sum[omp_get_thread_num()] = t;
  }

  double res = 0;
  for(uint32_t i = 0; i < num_threads; ++i)
  {
    res += err[i] + sum[i];
  }
  free(err);
  free(sum);
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
  uint32_t x_last = 0, num_threads = 0;
  input >> x_last >> num_threads;
  // Calculation
  double res = calc(x_last, num_threads);

  // Write result
  output << std::setprecision(15) << res << std::endl;
  // Prepare to exit
  output.close();
  input.close();
  return 0;
}
