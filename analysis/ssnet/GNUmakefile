
CC = g++
CFLAGS += -fPIC -g
INCLUDES += `larcv-config --includes`
INCLUDES += `root-config --cflags`
INCLUDES += `python-config --cflags`

LDFLAGS = -g -fPIC -lgomp -shared
LIBS = `larcv-config --libs`
LIBS += `root-config --libs`
LIBS += `python-config --libs`

all: libssnetdata.so test_read_loop

libssnetdata.so: SSNetDataLoader.o dict_ssnetdata.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

SSNetDataLoader.o: SSNetDataLoader.cxx SSNetDataLoader.h 
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

dict_ssnetdata.o: SSNetDataLoader.h
	rootcling -f $(basename $@).cxx -rml libssnetdata.so -c $(INCLUDES) $^ LinkDef.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $(basename $@).cxx -o $@

test_read_loop: test_read_loop.cxx libssnetdata.so
	$(CC) $(CFLAGS) $(INCLUDES) $@.cxx -o $@ libssnetdata.so $(LIBS) -lLArCVCorePyUtil

clean:
	rm *.o *.so
