#include "FillScoreHist.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include <sstream>

namespace ssnet {
  
  bool FillScoreHist::_setup_numpy = false;

  FillScoreHist::FillScoreHist()
    : larcv::larcv_base("FillScoreHist")
  {
    _class_name_v = { "background", "shower", "track" };
  }
  
  void FillScoreHist::define_hists()
  {
    int num_classes = _class_name_v.size();
    
    for (int iclass=0; iclass<num_classes; iclass++) {
      
      std::string classname = _class_name_v[iclass];

      // fill score values for pixels with max-score being of a certain class
      std::stringstream hname_class_scores;
      hname_class_scores << "hscores_per_max" << classname;
      std::stringstream htitle_class_scores;
      htitle_class_scores << ";scores of " << classname << " pixels; fraction of " << classname << " pixels";
      
      TH1D* hscores_per_class = new TH1D( hname_class_scores.str().c_str(),
					  htitle_class_scores.str().c_str(),
					  50,
					  0, 1.0 );

      _hscores_per_class_v.push_back( hscores_per_class );

      
      // fill number of pixels of certain classes per image
      std::stringstream hname_classpix_per_image;
      hname_classpix_per_image << "hnpix_per_image_" << classname;
      std::stringstream htitle_classpix_per_image;
      htitle_classpix_per_image << ";num " << classname << " pixels per image; fraction of images";
      
      TH1D* hclasspix_per_image = new TH1D( hname_classpix_per_image.str().c_str(),
					    htitle_classpix_per_image.str().c_str(),
					    50,
					    0, 500 );

      _hnpix_per_image_v.push_back( hclasspix_per_image );

      // scores above threshold
      std::stringstream hname_score_per_abovethresh;
      hname_score_per_abovethresh << "hscore_abovethresh_" << classname;
      std::stringstream htitle_score_per_abovethresh;
      htitle_score_per_abovethresh << ";" << classname << " scores for pixels above threshold; fraction of all pixels";
      
      TH1D* hscore_per_pixabovethresh = new TH1D( hname_score_per_abovethresh.str().c_str(),
						  htitle_score_per_abovethresh.str().c_str(),
						  50,
						  0, 1.0 );

      _hscores_above_thresh_v.push_back( hscore_per_pixabovethresh );
      
    }
  }
  
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

    // LARCV_NORMAL() << "image size: (" << img_dims[0] << "," << img_dims[1] << ")" << std::endl;
    // LARCV_NORMAL() << "score image size: (" << score_dims[0] << "," << score_dims[1] << "," << score_dims[2] << ")" << std::endl;
    // LARCV_NORMAL() << "number of class histograms: " << class_score_hist.size() << std::endl;
    
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


  int FillScoreHist::fillInternalHists( PyObject* np_img_array,
					PyObject* np_score_array,
					float img_thresh )
  {

    if ( !FillScoreHist::_setup_numpy ) {
      import_array1(0);
      FillScoreHist::_setup_numpy = true;
    }

    int num_classes = _class_name_v.size();    
    
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

    // LARCV_NORMAL() << "image size: (" << img_dims[0] << "," << img_dims[1] << ")" << std::endl;
    // LARCV_NORMAL() << "score image size: (" << score_dims[0] << "," << score_dims[1] << "," << score_dims[2] << ")" << std::endl;
    // LARCV_NORMAL() << "number of class histograms: " << class_score_hist.size() << std::endl;
    
    // if ( true )
    //   return 0;

    std::vector<int> num_classpix_v(3,0);    
    int nabove_thresh = 0;

    // loop over every pixel
    for (int r=0; r<(int)img_dims[0]; r++) {
      for (int c=0; c<(int)img_dims[1]; c++) {

	// is pixel value above threshold
	bool above_thresh = (img_carray[r][c]>img_thresh);

	// find max class score. if pixel is below threshold, it gets assigned a background score of 1.0
	int maxclass = 0;
	float maxscore = 0.;
	if ( above_thresh ) {
	  for (int iclass=0; iclass<3; iclass++) {	  
	    if ( score_carray[iclass][r][c]>maxscore )  {
	      maxclass = iclass;
	      maxscore = score_carray[iclass][r][c];
	    }
	  }
	  nabove_thresh++;	  
	}
	else {
	  maxscore = 0.999;
	  maxclass = 0;
	}

	// limit score value to avoid going out of bin range
	// (a ROOT quirk)
	if (maxscore>0.999)
	  maxscore = 0.999;
	
	_hscores_per_class_v[maxclass]->Fill( maxscore );

	if ( above_thresh ) {
	  num_classpix_v[maxclass]++;	  
	  for (int iclass=0; iclass<num_classes; iclass++) {
	    float class_score = score_carray[iclass][r][c];
	    if ( class_score>0.999 )
	      class_score = 0.999;
	    _hscores_above_thresh_v[iclass]->Fill( class_score );
	  }
	}
      }
    }
    
    for (int iclass=0; iclass<num_classes; iclass++) {
      _hnpix_per_image_v[iclass]->Fill( num_classpix_v[iclass] );
    }

    return nabove_thresh;
    
  }
  

}
