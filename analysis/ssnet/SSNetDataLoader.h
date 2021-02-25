#ifndef __SSNET_DATA_LOADER__
#define __SSNET_DATA_LOADER__

#include <Python.h>
#include "bytesobject.h"

#include <vector>
#include <string>

#include "TRandom3.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"

namespace ssnet {

  /** 
   * @class SSNetDataLoader
   * @ingroup SSNetDataLoader
   * @brief Load tensors from larcv2 root file for ssnet training and inference
   *
   */  
  class SSNetDataLoader : public larcv::larcv_base {

  public:

    SSNetDataLoader()
      : larcv::larcv_base("SSNetDataLoader"),
      _current_entry(0),
      _num_entries(0),
      _shuffle(false),
      _kowner(false),
      _adc_image_treename("data"),
      _truth_image_treename("segment"),
      _rand(nullptr)
      {};
    //SSNetDataLoader( const std::vector<std::string>& input_root_files ); 
    virtual ~SSNetDataLoader() {}; 

    void set_adc_image_treename( std::string name ) { _adc_image_treename=name; };
    void set_truth_image_treename( std::string name ) { _truth_image_treename=name; };    

    PyObject* makeTrainingDataDict( int batchsize );

    unsigned long   getCurrentEntry() { return _current_entry; };

  protected:

    larcv::IOManager _io;
    unsigned long _current_entry;
    unsigned long _num_entries;
    bool _shuffle; ///< if true, shuffle entry when loading batch
    bool _kowner;  ///< indicates if we own the tree (and must delete in destructor)
    int  _kMaxTripletPerVoxel; ///< maximum number of spacepoints contributing to voxel
    std::string _adc_image_treename;
    std::string _truth_image_treename;

    TRandom3* _rand;

  private:

    static bool _setup_numpy;
    
  };
  
}

#endif
