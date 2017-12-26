
#include <iostream>

#include <opencv2/opencv.hpp>

struct Voxel  // 4*5 = 20 bytes
{
        float sdf;
        float weight;
        float color[3];
};


__global__ void setZero_kernel1D (Voxel* voxel, const size_t size, const unsigned dimx, const unsigned dimy, const unsigned dimz) {
        size_t offset = threadIdx.x + blockIdx.x * blockDim.x;

        size_t step = gridDim.x * blockDim.x;

        while (offset < size)
        {
            voxel[offset].weight = 1;

            offset += step;
        }
}

__global__ void setZero_kernel (Voxel* voxel, const size_t size, const unsigned dimx, const unsigned dimy, const unsigned dimz) {
        size_t x = threadIdx.x + blockIdx.x * blockDim.x;
        size_t y = threadIdx.y + blockIdx.y * blockDim.y;
        size_t z = threadIdx.z + blockIdx.z * blockDim.z;

        size_t step = gridDim.x*blockDim.x*gridDim.y*blockDim.y*gridDim.z*blockDim.z;

        size_t offset =   z * gridDim.x*blockDim.x*gridDim.y*blockDim.y
                        + y * gridDim.x*blockDim.x
                        + x;

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
                //init (dim==0? 700: dim);
        }

        void init (int dim=500) {
                if (size != 0) cudaFree (voxel);
                dimx = dimy = dimz = dim;
                size = dimx*dimy*dimz;
                printf ("@ TSDF size = %.2f giga bytes.\n", (double)size * sizeof(Voxel) / (1024*1024*1024));
                printf ("@ sizeof (size_t) = %lu sizeof(Grid*)=%lu\n", sizeof (size_t), sizeof(Voxel*));
                //voxel = new Grid [s] ;
                cudaMallocManaged ((void**)&voxel, size*sizeof(Voxel));
                memset (voxel, 0, size*sizeof(Voxel)); // clear buffer

                printf ("@ Voxel allocated: %ux%ux%u\n", dimx, dimy, dimz);
        }

        ~TSDF () {
                if (voxel)
                    cudaFree (voxel);
        }

        void setZero() {
                if (0)
                {
                    printf ("-- calling 3D kernel\n");
                    dim3 block(1024,1,1);
                    dim3 grid(32,32,32);
                    setZero_kernel<<<grid,block>>> (voxel, size, dimx, dimy, dimz);
                }


                printf ("-- calling 1D kernel\n");
                dim3 nthreads(1024);
                dim3 nblocks(512);
                setZero_kernel1D <<<nblocks,nthreads>>> (voxel, size, dimx, dimy, dimz);

                //__global__ void setZero_kernel (TSDF& tsdf); 
                //setZero_kernel<<<grid,block>>> (*this);

                cudaDeviceSynchronize ();
        }
};


__global__ void setZero_kernel (TSDF& tsdf) {
        size_t x = threadIdx.x + blockIdx.x * blockDim.x;
        size_t y = threadIdx.y + blockIdx.y * blockDim.y;
        size_t z = threadIdx.z + blockIdx.z * blockDim.z;

        size_t step = gridDim.x*blockDim.x*gridDim.y*blockDim.y*gridDim.z*blockDim.z;

        size_t offset =   z * gridDim.x*blockDim.x*gridDim.y*blockDim.y
                        + y * gridDim.x*blockDim.x
                        + x;

        while (offset < tsdf.size)
        {
            tsdf.voxel[offset].weight = 1;

            offset += step;
        }
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


// ----------------------------------------------------------------------------------------------

TSDF tsdf;

int main() {

        cudaInfo();

        //cudaMallocManaged (&ptsdf, sizeof(TSDF));
        //TSDF &tsdf = *ptsdf;

        tsdf.init();

        //for (size_t i=0; i<tsdf.size; i++)
        //    printf ("@ weight[%lu] = %.1f\n", i, tsdf.voxel[i].weight);

        printf("@ setZero() test.\n");
        for (int i=0; i<10; i++)
        tsdf.setZero();

        //for (size_t i=0; i<tsdf.size; i++)
        //    printf ("@ weight[%lu] = %.1f\n", i, tsdf.voxel[i].weight);

        bool flag = true;
        for (size_t i=0; i<tsdf.size; i++)
                if (tsdf.voxel[i].weight != 1) {
                        printf ("@ non zero found at %lu\n", i), flag=false;
                        break;
                }
        printf ("@ test result is %d.\n", flag);

        //
        // 1. read a depth image and integrate it into tsdf
        //



        cv::waitKey ();

        cudaDeviceReset ();
        return 0;
}

// eof 
