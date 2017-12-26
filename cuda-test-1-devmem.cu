
#include <iostream>

#include <opencv2/opencv.hpp>

struct Voxel  // 4*5 = 20 bytes
{
        float sdf;
        float weight;
        float color[3];
};


__global__ void setZero_kernel (Voxel* voxel, const size_t size, const unsigned dimx, const unsigned dimy, const unsigned dimz) {
        size_t offset = threadIdx.x + blockIdx.x * blockDim.x;

        size_t step = gridDim.x*blockDim.x;

        while (offset < size)
        {
            voxel[offset].weight = 1;

            offset += step;
        }
}


struct TSDF 
{
        Voxel *voxel;
        unsigned dimx, dimy, dimz;
        size_t size;

        TSDF(int dim=0) {
                size = 0;
        }

        void init (int dim=500) {
                if (size != 0) cudaFree (voxel);
                dimx = dimy = dimz = dim;
                size = dimx*dimy*dimz;
                printf ("@ TSDF size = %.2f giga bytes.\n", (double)size * sizeof(Voxel) / (1024*1024*1024));
                printf ("@ sizeof (size_t) = %lu sizeof(Grid*)=%lu\n", sizeof (size_t), sizeof(Voxel*));
                cudaMalloc (&voxel, sizeof(Voxel)*size);
                cudaMemset (&voxel, 0, size*sizeof(Voxel)); // clear buffer

                printf ("@ Voxel allocated: %ux%ux%u\n", dimx, dimy, dimz);
        }

        ~TSDF () {
                if (voxel)
                    cudaFree (voxel);
        }

        void setZero() {
                dim3 block(1024);
                dim3 grid(512);

                setZero_kernel<<<grid,block>>> (voxel, size, dimx, dimy, dimz);

                cudaDeviceSynchronize ();
        }
};


TSDF tsdf;

void cudaInfo ();

int main() {

        cudaInfo();

        tsdf.init();

        //for (size_t i=0; i<tsdf.size; i++)
        //    printf ("@ weight[%lu] = %.1f\n", i, tsdf.voxel[i].weight);

        printf("@ setZero() test.\n");
        for (int i=0; i<10; i++)
        tsdf.setZero();

        //for (size_t i=0; i<tsdf.size; i++)
        //    printf ("@ weight[%lu] = %.1f\n", i, tsdf.voxel[i].weight);

        bool flag = true;
        if (0)
        for (size_t i=0; i<tsdf.size; i++)
                if (tsdf.voxel[i].weight != 1)
                        printf ("@ non zero found at %lu\n", i), flag=false;
        printf ("@ test result is %d.\n", flag);

        // 1. read a depth image and integrate it into tsdf



        cv::waitKey ();

        cudaDeviceReset ();
        return 0;
}

void cudaInfo ()
{
        const unsigned long gb = 1024*1024*1024;
        cudaSetDevice (0);

        size_t free_device_mem = 0;
        size_t total_device_mem = 0;

        cudaMemGetInfo (&free_device_mem, &total_device_mem);
        printf("Currently available amount of device memory: %.2f giga bytes\n", free_device_mem/(double)gb);
        printf("Total amount of device memory: %.2f giga bytes\n", total_device_mem/(double)gb);
}
