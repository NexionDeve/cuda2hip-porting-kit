CUDA_SRC = examples/saxpy.cu
HIP_SRC = examples/saxpy.cpp
CUDA_BIN = saxpy_cuda
HIP_BIN = saxpy_hip

.PHONY: cuda hip clean

cuda:
	nvcc $(CUDA_SRC) -o $(CUDA_BIN)

hip:
	hipcc $(HIP_SRC) -o $(HIP_BIN)

clean:
	rm -f $(CUDA_BIN) $(HIP_BIN) *.o *.out
