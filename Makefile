CC = nvcc -g  -x cu -arch compute_52
CVLIBS= -lopencv_imgcodecs -lopencv_core -lopencv_highgui 

all: tsdf.cpp
	$(CC) -std=c++11 tsdf.cpp -I/usr/local/cuda/include -lcudart -lcublas -lcurand $(CVLIBS)


test:
	$(CC) -std=c++11 glm-test.cu
	$(CC) -std=c++11 tsdf-kernel-test.cu -I/usr/local/cuda/include -lcudart -lcublas -lcurand $(CVLIBS)
