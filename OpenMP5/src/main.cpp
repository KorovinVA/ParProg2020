#include <iostream>
#include <iomanip>
#include <fstream>
#include <omp.h>

uint8_t isAlive(uint8_t* field, int x, int y, uint32_t xSize, uint32_t ySize)
{
  if(x < 0)
  {
    x = xSize - 1;
  }
  if(y < 0)
  {
    y = ySize - 1;
  }
  if (x == (int)xSize)
  {
    x = 0;
  }
  if( y == (int)ySize)
  {
    y = 0;
  }

  return field[y * xSize + x];
}

uint8_t proceed(uint8_t cell, uint8_t* field, uint32_t x, uint32_t y, uint32_t xSize, uint32_t ySize)
{
  uint8_t sum = 0;
  sum += isAlive(field, int(x - 1), int(y - 1), xSize, ySize);
  sum += isAlive(field, int(x - 1), int(y),     xSize, ySize);
  sum += isAlive(field, int(x - 1), int(y + 1), xSize, ySize);
  sum += isAlive(field, int(x),     int(y - 1), xSize, ySize);
  sum += isAlive(field, int(x),     int(y + 1), xSize, ySize);
  sum += isAlive(field, int(x + 1), int(y - 1), xSize, ySize);
  sum += isAlive(field, int(x + 1), int(y),     xSize, ySize);
  sum += isAlive(field, int(x + 1), int(y + 1), xSize, ySize);

  uint8_t newCell = 0;
  if(cell == 0)
  {
    if(sum == 3)
    {
      newCell = 1;
    }
  }
  else
  {
    if(sum == 2 || sum == 3)
    {
      newCell = 1;
    }
  }
  return newCell;
}

void calc(uint32_t xSize, uint32_t ySize, uint32_t iterations, uint32_t num_threads, uint8_t* inputFrame, uint8_t* outputFrame)
{

  for(uint32_t y = 0; y < ySize; ++y)
  {
    for(uint32_t x = 0; x < xSize; ++x)
    {
      outputFrame[y * xSize + x] = inputFrame[y * xSize + x];
    }
  }

  for(uint32_t i = 0; i < iterations; ++i)
  {
    #pragma omp parallel for num_threads(num_threads)
    for(uint32_t y = 0; y < ySize; ++y)
    {
      for(uint32_t x = 0; x < xSize; ++x)
      {
        outputFrame[y * xSize + x] = proceed(inputFrame[y * xSize + x],
                                             inputFrame,
                                             x,
                                             y,
                                             xSize,
                                             ySize);
      }
    }

    for(uint32_t y = 0; y < ySize; ++y)
    {
      for(uint32_t x = 0; x < xSize; ++x)
      {
        inputFrame[y * xSize + x] = outputFrame[y * xSize + x];
      }
    }
  }
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
  uint32_t xSize = 0, ySize = 0, iterations = 0, num_threads = 0;
  input >> xSize >> ySize >> iterations >> num_threads;
  uint8_t* inputFrame = new uint8_t[xSize*ySize];
  uint8_t* outputFrame = new uint8_t[xSize*ySize];
  for (uint32_t y = 0; y < ySize; y++)
  {
    for (uint32_t x = 0; x < xSize; x++)
    {
      input >> inputFrame[y*xSize + x];
      inputFrame[y*xSize + x] -= '0';
    }
  }

  // Calculation
  calc(xSize, ySize, iterations, num_threads, inputFrame, outputFrame);

  // Write result
  for (uint32_t y = 0; y < ySize; y++)
  {
    for (uint32_t x = 0; x < xSize; x++)
    {
      outputFrame[y*xSize + x] += '0';
      output << " " << outputFrame[y*xSize + x];
    }
    output << "\n";
  }

  // Prepare to exit
  delete outputFrame;
  delete inputFrame;
  output.close();
  input.close();
  return 0;
}
