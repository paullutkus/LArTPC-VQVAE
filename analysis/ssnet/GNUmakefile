
CC = g++
CFLAGS += -fPIC -g
INCLUDES += `larcv-config --includes`
INCLUDES += `root-config --cflags`
INCLUDES += `python-config --cflags`

LDFLAGS = -g -fPIC -lgomp -shared
LIBS = `larcv-config --libs`
LIBS += `root-config --libs`
LIBS += `python-config --libs`

all: libssnetdata.so

libssnetdata.so: SSNetDataLoader.o dict_ssnetdata.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

SSNetDataLoader.o: SSNetDataLoader.cxx SSNetDataLoader.h 
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

dict_ssnetdata.o: SSNetDataLoader.h
	rootcling -f $(basename $@).cxx -rml libssnetdata.so -c $(INCLUDES) $^ LinkDef.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $(basename $@).cxx -o $@


