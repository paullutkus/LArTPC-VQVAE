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
      _io(nullptr),
      _current_entry(0),
      _num_entries(0),
      _shuffle(false),
      _kowner(false),
      _adc_image_treename("data"),
      _truth_image_treename("segment"),
      _colsize(0),
      _rowsize(0),
      _numplanes(0),
      _rand(nullptr),
      _timer_alloc(0),
      _timer_ioman_read(0),
      _timer_datacopy(0)
      {};
    //SSNetDataLoader( const std::vector<std::string>& input_root_files ); 
    virtual ~SSNetDataLoader(); 

    void setup( std::string data_file_path, int ran_seed = 0, bool with_truth=true );
    void set_adc_image_treename( std::string name ) { _adc_image_treename=name; };
    void set_truth_image_treename( std::string name ) { _truth_image_treename=name; };    

    PyObject* makeTrainingDataDict( int batchsize, int plane=-1 );

    unsigned long   getCurrentEntry() { return _current_entry; };

    float getAllocTime() { return _timer_alloc; };
    float getReadTime() { return _timer_ioman_read; };
    float getCopyTime() { return _timer_datacopy; };
    float getTotalTime() { return _timer_alloc+_timer_ioman_read+_timer_datacopy; };
    void resetTimers() { _timer_alloc=_timer_ioman_read=_timer_datacopy=0.0; };

  protected:

    larcv::IOManager* _io;
    unsigned long _current_entry;
    unsigned long _num_entries;
    bool _shuffle; ///< if true, shuffle entry when loading batch
    bool _kowner;  ///< indicates if we own the tree (and must delete in destructor)
    int  _kMaxTripletPerVoxel; ///< maximum number of spacepoints contributing to voxel
    std::string _adc_image_treename;
    std::string _truth_image_treename;
    int _colsize;
    int _rowsize;
    int _numplanes;

    TRandom3* _rand;

    float _timer_alloc;
    float _timer_ioman_read;
    float _timer_datacopy;

  private:

    static bool _setup_numpy;
    
  };
  
}

#endif
