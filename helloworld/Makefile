HIP_PATH?= $(wildcard /opt/rocm)
HIPCC=$(HIP_PATH)/bin/hipcc

SOURCES = HelloWorld.cc
OBJECTS = HelloWorld.o
TARGET=HelloWorld




CPPFLAGS=-std=c++17 -O3 -Wall -march=native -mavx2 -mno-avx512f -mfma -fopenmp --offload-arch=gfx908
# CXXFLAGS =-g
CXX=$(HIPCC)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(HIPCC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS)


clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
	rm -f $(HIP_PATH)/src/*.o