#include "FillScoreHist.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

namespace ssnet {
  
  bool FillScoreHist::_setup_numpy = false;
  
  int FillScoreHist::fillHist( PyObject* np_img_array,
                               PyObject* np_score_array,
                               std::vector<TH1D>& class_score_hist,
                               float img_thresh )
  {

    if ( !FillScoreHist::_setup_numpy ) {
      import_array1(0);
      FillScoreHist::_setup_numpy = true;
    }
    
    PyArray_Descr *descr_float  = PyArray_DescrFromType(NPY_FLOAT);

    npy_intp img_dims[2];
    float **img_carray;
    if ( PyArray_AsCArray( &np_img_array, (void**)&img_carray, img_dims, 2, descr_float )<0 ) {
      LARCV_CRITICAL() << "Cannot get carray for output img tensor" << std::endl;
      throw std::runtime_error("Cannot get carray for output img tensor");
    }
    
    npy_intp score_dims[3];
    float ***score_carray;
    if ( PyArray_AsCArray( &np_score_array, (void***)&score_carray, score_dims, 3, descr_float )<0 ) {
      LARCV_CRITICAL() << "Cannot get carray for output score tensor" << std::endl;
      throw std::runtime_error("Cannot get carray for output score tensor");
    }

    LARCV_NORMAL() << "image size: (" << img_dims[0] << "," << img_dims[1] << ")" << std::endl;
    LARCV_NORMAL() << "score image size: (" << score_dims[0] << "," << score_dims[1] << "," << score_dims[2] << ")" << std::endl;
    LARCV_NORMAL() << "number of class histograms: " << class_score_hist.size() << std::endl;
    
    // if ( true )
    //   return 0;

    int nabove_thresh = 0;
    for (int r=0; r<(int)img_dims[0]; r++) {
      for (int c=0; c<(int)img_dims[1]; c++) {

        if ( img_carray[r][c]<img_thresh ) continue;
        
        for (int iclass=0; iclass<(int)class_score_hist.size(); iclass++) {
          class_score_hist[iclass].Fill( score_carray[iclass][r][c] );
        }

        nabove_thresh++;
      }
    }

    return nabove_thresh;
    
  }

}
