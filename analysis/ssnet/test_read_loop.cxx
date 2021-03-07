#include <iostream>
#include <time.h>

//#include <Python/Python.h>
#include <Python.h>

#include "larcv/core/PyUtil/PyUtils.h"
#include "SSNetDataLoader.h"

int main( int nargs, char** argv ) {
  
  std::cout << "Test read loop speed" << std::endl;
  std::clock_t start = std::clock();

  Py_Initialize();
  
  larcv::SetPyUtil();


  
  int batchsize = 16;
  int nbatches = 100;

  if ( nargs==2 )
    nbatches = std::atoi(argv[1]);
  if ( nargs==3 )
    batchsize = std::atoi(argv[2]);

  std::cout << "Number of batches to run: " << nbatches << std::endl;
  std::cout << " batchsize: " << batchsize << std::endl;
  
  ssnet::SSNetDataLoader loader;
  loader.setup( "test_10k.root", 0, true );
  for (int i=0; i<nbatches; i++) {
    PyObject* dict = loader.makeTrainingDataDict( batchsize, 2 );
    Py_DECREF( dict );
  }

  float end = float( std::clock()-start )/CLOCKS_PER_SEC;
  std::cout << "end: " << end << std::endl;


  float dt_copy = loader.getCopyTime();
  float dt_read = loader.getReadTime();
  float dt_alloc = loader.getAllocTime();

  std::cout << "total time elapsed: " << end << " secs" << std::endl;
  std::cout << "  dt_alloc: " << dt_alloc << std::endl;
  std::cout << "  dt_read:  " << dt_read << std::endl;
  std::cout << "  dt_copy:  " << dt_copy << std::endl;
  std::cout << std::endl;
  std::cout << "total time per iteration: " << end/(nbatches*batchsize) << " sec/iter" << std::endl;
  std::cout << "  dt_alloc: " << dt_alloc/(nbatches*batchsize) << std::endl;
  std::cout << "  dt_read:  " << dt_read/(nbatches*batchsize) << std::endl;
  std::cout << "  dt_copy:  " << dt_copy/(nbatches*batchsize) << std::endl;
  
  std::cout << "done" << std::endl;
  return 0;
  
}
