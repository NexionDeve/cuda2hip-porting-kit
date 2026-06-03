NVCC = nvcc
HIPCC = hipcc
BUILD_DIR = build
NVCC_FLAGS = -O2
HIPCC_FLAGS = -O2

CUDA_SRCS = $(wildcard examples/*.cu)
HIP_SRCS = $(wildcard examples/*.cpp)
CUDA_BINS = $(patsubst examples/%.cu,$(BUILD_DIR)/%_cuda,$(CUDA_SRCS))
HIP_BINS = $(patsubst examples/%.cpp,$(BUILD_DIR)/%_hip,$(HIP_SRCS))

.PHONY: all cuda hip clean

all: cuda hip

cuda: $(CUDA_BINS)

hip: $(HIP_BINS)

$(BUILD_DIR)/%_cuda: examples/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

$(BUILD_DIR)/%_hip: examples/%.cpp | $(BUILD_DIR)
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)
