PROJECT = cuda_app
NVCC = nvcc
SRCS = src/main.cu src/kernels.cu
OBJS = $(SRCS:.cu=.o) 
INCLUDES = -I./include
ARCH = --gpu-architecture=sm_75
FLAGS = -Xcompiler "/W3 /O2"

all: $(PROJECT)

$(PROJECT): $(OBJS)
	$(NVCC) $(ARCH) $(FLAGS) $(INCLUDES) -o $@ $^


%.o: %.cu
	$(NVCC) $(ARCH) $(FLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(PROJECT) $(OBJS)

.PHONY: all clean