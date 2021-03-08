#ifndef __SSNET_FILL_SCORE_HIST_H__
#define __SSNET_FILL_SCORE_HIST_H__

#include <Python.h>
#include "bytesobject.h"

#include <vector>
#include "TH1D.h"
#include "larcv/core/Base/larcv_base.h"


namespace ssnet {

  class FillScoreHist : public larcv::larcv_base {

  public:

    FillScoreHist()
      : larcv::larcv_base("FillScoreHist")
      {};
    virtual ~FillScoreHist() {};

    int fillHist( PyObject* np_img_array,
                  PyObject* np_score_array,
                  std::vector<TH1D>& class_score_hist,
                  float img_thresh );

  protected:

    static bool _setup_numpy;
    
  };

}

#endif
