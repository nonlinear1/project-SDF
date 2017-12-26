
all:
	nvcc -std=c++11 tsdf.cu -I/usr/local/cuda/include -lcudart -lcublas -lcurand -lopencv_core -lopencv_highgui -Wno-deprecated-gpu-targets
