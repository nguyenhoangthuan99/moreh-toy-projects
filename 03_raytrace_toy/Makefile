CC=gcc
CXX=hipcc
INCLUDES=
CFLAGS=-Wall --offload-arch=gfx908 -mfma -fopenmp
LDFLAGS=-lm -ljpeg -fopenmp
TARGET=raytracer
OBJS=raytracer.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS) 

.c.o:
	$(CC) $(CFLAGS) -c $< -o $@
.cc.o:
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(TARGET) $(OBJS)

run: $(TARGET)
	./$(TARGET)
