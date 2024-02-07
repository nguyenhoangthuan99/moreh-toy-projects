HIP_PATH?= $(wildcard /opt/rocm)
HIPCC=$(HIP_PATH)/bin/hipcc
CXX=$(HIPCC)
LD=$(HIPCC)
# CFLAGS  = -O3 -Wall -g -DSAVE_JPG  
CPPFLAGS=-std=c++17 -O3 -Wall -DSAVE_JPG  -fopenmp --offload-arch=gfx908
LDFLAGS	= -lm -ljpeg

EXE = fractals
CXXFILES	=	$(wildcard *.cc)
OBJECTS	=	$(CXXFILES:.cc=.o)

all: $(EXE)

$(EXE) : $(OBJECTS)
	$(LD) -o $@ $^ $(CPPFLAGS) $(LDFLAGS) 

%.o : %.cc
	$(CXX) -c $< -o $@  $(CPPFLAGS)

clean :	
	rm -f $(EXE) $(OBJECTS) *.ppm
