#include <iostream>
#include <iomanip>
#include <fstream>

#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include <unistd.h>
#include <cmath>
#include <cstring>

double acceleration(double t)
{
  return sin(t);
}

void calc(double* trace, uint32_t traceSize, double t0, double dt, double y0, double y1, int rank, int size)
{
  double v0 = 0;
  MPI_Bcast(&traceSize, 1, MPI_INT,    0, MPI_COMM_WORLD);
  MPI_Bcast(    &y0   , 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(    &y1   , 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(    &t0   , 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(    &dt   , 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  uint32_t regionSize    = traceSize / (uint32_t)size;
  uint32_t regionSizeLTh = regionSize + traceSize % (uint32_t) size;      //Size of a region for the first proc
  uint32_t rSize = (rank == 0 ? regionSizeLTh : regionSize);

  y0 = (rank == 0 ? y0 : 0);
  t0 += rSize * rank * dt;
  double* buf  = new double[rSize];

  // Calculate trajectory for each region. Kahan algorithm
  buf[0] = y0;
  buf[1] = y0 + dt * v0;
  for (uint32_t i = 2; i < rSize; i++)
  {
    buf[i] = dt * dt * acceleration(t0 + (i - 1) * dt) +
              2 * buf[i - 1] - buf[i - 2];
  }
  double lastV  = (buf[rSize - 1] - buf[rSize - 2]) / dt;
  double lastY  = buf[rSize - 1];
  //double lastY = dt * dt * acceleration(t0 + (rSize - 1) * dt) +
              //2 * buf[rSize - 1] - buf[rSize - 2];

  //save V and Y; restore before the second alignment
  double savedV = lastV;
  double savedY = lastY;

  if(size == 1)
  {
    v0 = (y1 - lastY) / (traceSize * dt);
    std::cout << v0 << std::endl;
  }
  else
  {
    //First y and v alignment
    if(rank != 0)
    {
      MPI_Recv(&buf[0], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(  &v0  , 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      lastY += buf[0] + v0 * dt * rSize;
      lastV += v0;
    }
    if(rank != size - 1)
    {
      MPI_Send(&lastY, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
      MPI_Send(&lastV, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
    }

    //Send real v0 to proc0
    if(rank == size - 1)
    {
      double bStar = lastY;
      double vStar = (y1 - bStar) / (dt * traceSize);
      MPI_Send(&vStar, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    else if(rank == 0)
    {
      MPI_Recv(&v0, 1, MPI_DOUBLE, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    //Second y and v alignment(have v0 now)
    lastV = savedV;
    lastY = savedY;

    if(rank != 0)
    {
      MPI_Recv(&buf[0], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(  &v0  , 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      lastY += buf[0];
    }
    if(rank != size - 1)
    {
      lastV += v0;
      lastY += v0 * dt * rSize;
      MPI_Send(&lastY, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
      MPI_Send(&lastV, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
    }
  }

  //Final shot
  buf[1] = y0 + dt * v0;
  for (uint32_t i = 2; i < rSize; i++)
  {
    buf[i] = dt * dt * acceleration(t0 + (i - 1) * dt) +
              2 * buf[i - 1] - buf[i - 2];
  }

  //Save data to trace array
  if(rank == 0)
  {
    for(int i = 0; i < size; i++)
    {
      if(i == 0)
      {
        std::memcpy(trace, buf, rSize * sizeof(double));
      }
      else
      {
        MPI_Recv(&trace[i * regionSize + traceSize % (uint32_t) size], regionSize,
                  MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }
  }
  else
  {
    MPI_Send(buf, regionSize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }

  delete [] buf;
}

int main(int argc, char** argv)
{
  int rank = 0, size = 0, status = 0;
  uint32_t traceSize = 0;
  double t0 = 0, t1 = 0, dt = 0, y0 = 0, y1 = 0;
  double* trace = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0)
  {
    // Check arguments
    if (argc != 3)
    {
      std::cout << "[Error] Usage <inputfile> <output file>\n";
      status = 1;
      MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return 1;
    }

    // Prepare input file
    std::ifstream input(argv[1]);
    if (!input.is_open())
    {
      std::cout << "[Error] Can't open " << argv[1] << " for write\n";
      status = 1;
      MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return 1;
    }

    // Read arguments from input
    input >> t0 >> t1 >> dt >> y0 >> y1;
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    traceSize = (t1 - t0)/dt;
    trace = new double[traceSize];

    input.close();
  } else {
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (status != 0)
    {
      return 1;
    }
  }

  calc(trace, traceSize, t0, dt, y0, y1, rank, size);

  if (rank == 0)
  {
    // Prepare output file
    std::ofstream output(argv[2]);
    if (!output.is_open())
    {
      std::cout << "[Error] Can't open " << argv[2] << " for read\n";
      delete trace;
      return 1;
    }

    for (uint32_t i = 0; i < traceSize; i++)
    {
      output << " " << trace[i];
    }
    output << std::endl;
    output.close();
    delete trace;
  }

  MPI_Finalize();
  return 0;
}
