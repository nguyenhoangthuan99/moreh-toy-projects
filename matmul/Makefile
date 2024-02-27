CC=hipcc
CXX=hipcc
INCLUDES=
CFLAGS=-O3 -Wall --offload-arch=gfx908 -funroll-loops
LDFLAGS=-ljpeg -lm
TARGET=edge
OBJS=edge.o
CPPFLAGS=-std=c++17 -O3 -Wall --offload-arch=gfx908 -funroll-loops

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS) 

.c.o:
	$(CC) $(CFLAGS) -c $< -o $@

.cc.o:
	$(CXX) -c $< -o $@ $(CPPFLAGS)

clean:
	rm -rf $(TARGET) $(OBJS)

run: $(TARGET)
	./$(TARGET)
