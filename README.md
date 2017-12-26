# project-SDF
TSDF, CUDA, Fusion, Dynamic Object

1. TSDF, UM, Depth Image, Ray-casting




##### Learning Cuda
`cuda-test-1.cu` & `cuda-test-1-devmem.cu`
- It seems total block size <= 1024 in my Titan X.
- Global memory access test showed UM and dev mem have almost the same read-modify-write access time.

