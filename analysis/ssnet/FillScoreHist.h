#ifndef __SSNET_FILL_SCORE_HIST_H__
#define __SSNET_FILL_SCORE_HIST_H__

#include <Python.h>
#include "bytesobject.h"

#include <vector>
#include <string>
#include "TH1D.h"
#include "larcv/core/Base/larcv_base.h"


namespace ssnet {

  class FillScoreHist : public larcv::larcv_base {

  public:

    FillScoreHist();
    virtual ~FillScoreHist() {};

    void define_hists();
    
    int fillHist( PyObject* np_img_array,
                  PyObject* np_score_array,
                  std::vector<TH1D>& class_score_hist,
                  float img_thresh );

    int fillInternalHists( PyObject* np_img_array,
			   PyObject* np_score_array,
			   float img_thresh );

    std::vector< std::string > _class_name_v;
    std::vector< TH1D* > _hscores_per_class_v;
    std::vector< TH1D* > _hnpix_per_image_v;
    std::vector< TH1D* > _hscores_above_thresh_v;
    
    
  protected:

    static bool _setup_numpy;

    
  };

}

#endif
