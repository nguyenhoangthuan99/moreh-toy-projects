CC=gcc
CXX=hipcc
INCLUDES=
CFLAGS=-O2 -Wall
LDFLAGS=-ljpeg
TARGET=edge
OBJS=edge.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS) 

.c.o:
	$(CC) $(CFLAGS) -c $< -o $@

.cc.o:
	$(CXX) --offload-arch=gfx908 -c $< -o $@ $(CFLAGS)

clean:
	rm -rf $(TARGET) $(OBJS)

run: $(TARGET)
	./$(TARGET)
