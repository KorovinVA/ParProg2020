#include <iostream>
#include <iomanip>
#include <fstream>

#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include <unistd.h>
#include <cmath>

void calc(double* arr, uint32_t ySize, uint32_t xSize, int rank, int size)
{
  MPI_Bcast(&xSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&ySize, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0 && size == 1)
  {
    for (uint32_t y = 0; y < ySize; y++)
    {
      for (uint32_t x = 0; x < xSize; x++)
      {
        arr[y*xSize + x] = sin(0.00001*arr[y*xSize + x]);
      }
    }
  }
  else
  {
    uint32_t bufSize = ySize * xSize / size;

    double* buf = new double [bufSize];
    MPI_Scatter(arr, bufSize, MPI_DOUBLE, buf, bufSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for(uint32_t i = 0; i < bufSize; ++i)
    {
      buf[i] = sin(0.00001*buf[i]);
    }
    if(rank == 0)
    {
      for(uint32_t i = 1; i <= (ySize * xSize) % size; ++i)
      {
        arr[ySize * xSize - i] = sin(0.00001*arr[ySize * xSize - i]);
      }
    }
    MPI_Gather(buf, bufSize, MPI_DOUBLE, arr, bufSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    delete [] buf;
  }
}

int main(int argc, char** argv)
{
  int rank = 0, size = 0, buf = 0;
  uint32_t ySize = 0, xSize = 0;
  double* arr = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0)
  {
    // Check arguments
    if (argc != 3)
    {
      std::cout << "[Error] Usage <inputfile> <output file>\n";
      buf = 1;
      MPI_Bcast(&buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return 1;
    }

    // Prepare input file
    std::ifstream input(argv[1]);
    if (!input.is_open())
    {
      std::cout << "[Error] Can't open " << argv[1] << " for write\n";
      buf = 1;
      MPI_Bcast(&buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return 1;
    }

    // Read arguments from input
    input >> ySize >> xSize;
    MPI_Bcast(&buf, 1, MPI_INT, 0, MPI_COMM_WORLD);

    arr = new double[ySize * xSize];

    for (uint32_t y = 0; y < ySize; y++)
    {
     for (uint32_t x = 0; x < xSize; x++)
      {
        input >> arr[y*xSize + x];
      }
    }
    input.close();
  } else {
    MPI_Bcast(&buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (buf != 0)
    {
      return 1;
    }
  }

  calc(arr, ySize, xSize, rank, size);

  if (rank == 0)
  {
    // Prepare output file
    std::ofstream output(argv[2]);
    if (!output.is_open())
    {
      std::cout << "[Error] Can't open " << argv[2] << " for read\n";
      delete arr;
      return 1;
    }
    for (uint32_t y = 0; y < ySize; y++)
    {
      for (uint32_t x = 0; x < xSize; x++)
      {
        output << " " << arr[y*xSize + x];
      }
      output << std::endl;
    }
    output.close();
    delete arr;
  }

  MPI_Finalize();
  return 0;
}
