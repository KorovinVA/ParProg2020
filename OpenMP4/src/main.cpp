#include <iostream>
#include <iomanip>
#include <fstream>
#include <omp.h>
#include <vector>

double fact(uint32_t num)
{
  double res = 1;
  #pragma omp parallel for reduction(*:res)
  for(uint32_t i = 1; i <= num; ++i)
  {
    res *= (double)i;
  }
  return res;
}

double calc(uint32_t x_last, uint32_t num_threads)
{
  std::vector<double> sum(num_threads);
  std::vector<double> err(num_threads);
  const uint32_t maxFact = 170;
  omp_set_nested(1);

  #pragma omp parallel for num_threads(num_threads)
  for(uint32_t i = 0 ; i < x_last; ++i)
  {
    //No reason to add numbers < 1/maxFact!.
    //Their values are less than double precision
    if(i <= maxFact)
    {
      double input = 1 / fact(i);
      double t     = sum[omp_get_thread_num()] + input;

      err[omp_get_thread_num()] += sum[omp_get_thread_num()] - t + input;
      sum[omp_get_thread_num()] = t;
    }
  }

  double res = 0;
  for(uint32_t i = 0; i < num_threads; i++)
  {
    res += sum[i] + err[i];
  }
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
  output << std::setprecision(16) << res << std::endl;
  // Prepare to exit
  output.close();
  input.close();
  return 0;
}
