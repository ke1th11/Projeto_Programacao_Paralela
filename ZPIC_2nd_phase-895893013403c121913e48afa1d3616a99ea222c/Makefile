CC = nvcc
CFLAGS = -O2 -g -std=c++11
# these are for search k20 GPU

#NCCFLAGS = -Wno-deprecated-gpu-targets -arch=sm_35 -O2 -g -std=c++11
NCCFLAGS = -Wno-deprecated-gpu-targets -arch=sm_80 -O2 -g -std=c++11
LDFLAGS = -lm
LINKER := nvcc

C_SOURCES := $(wildcard *.c)
CU_SOURCES := $(wildcard *.cu)
HEADERS := $(wildcard *.h)
C_OBJS := $(patsubst %.c, %.o, $(C_SOURCES))
CU_OBJS := $(patsubst %.cu, %.o, $(CU_SOURCES))

BIN = zpic

$(BIN) : $(C_OBJS) $(CU_OBJS) $(HEADERS)
	$(LINKER) -o $(BIN) $(CU_OBJS) $(C_OBJS) $(LDFLAGS) $(LIBS)

$(C_OBJS): $(C_SOURCES) $(HEADERS)
	$(CC) -c $(C_SOURCES) $(CFLAGS)

$(CU_OBJS): $(CU_SOURCES) $(HEADERS)
	$(CC) -c $(CU_SOURCES) $(NCCFLAGS)

run: $(BIN)
	sbatch run.sh

clean:
	rm -f $(BIN) *.o
