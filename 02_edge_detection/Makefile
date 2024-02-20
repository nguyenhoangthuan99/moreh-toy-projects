CC=hipcc
CXX=hipcc
INCLUDES=
CFLAGS=-O3 -Wall
LDFLAGS=-ljpeg -lm
TARGET=edge
OBJS=edge.o
CPPFLAGS=-std=c++17 -O3 -Wall

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS) 

.c.o:
	$(CC) $(CFLAGS) -c $< -o $@

.cc.o:
	$(CXX) --offload-arch=gfx908 -c $< -o $@ $(CPPFLAGS)

clean:
	rm -rf $(TARGET) $(OBJS)

run: $(TARGET)
	./$(TARGET)
