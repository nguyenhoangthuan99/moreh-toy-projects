CXX = g++
LD = g++
CFLAGS  = -O2 -Wall -g -DSAVE_JPG
LDFLAGS	= -lm -ljpeg

EXE = fractals
CXXFILES	=	$(wildcard *.cc)
OBJECTS	=	$(CXXFILES:.cc=.o)

all: $(EXE)

$(EXE) : $(OBJECTS)
	$(LD) -o $@ $^ $(CFLAGS) $(LDFLAGS) 

%.o : %.cc
	$(CXX) -c $< -o $@  $(CFLAGS)

clean :	
	rm -f $(EXE) $(OBJECTS) *.ppm
