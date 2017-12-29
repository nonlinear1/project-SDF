
#include <vector>
#include <tuple>
#include <utility>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;


#define GB (1024*1024*1024)


//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {						\
    cudaError_t e=cudaGetLastError();					\
    if(e!=cudaSuccess) {						\
      printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
      exit(0);								\
    }									\
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

inline  int divup(size_t a, size_t b) {
    return (a+b-1)/b;
}

// pre-declarations for function prototypes
//
struct TSDF;
template<typename T> struct cuImage;
struct Camera;

__global__ void kernel_integrate (TSDF& tsdf, cuImage<float>& depth, Camera& camera);
__global__ void kernel_setZero (TSDF& tsdf); 

/////////////////////////////////////////////////////////////////////////////////////
struct RGBZData {
cv::Mat_<float> zimg;
cv::Mat_<cv::Vec3b> rgb;

  void loadTUMSampleData()
// load a test depth image (TUM datafile)
// https://vision.in.tum.de/data/datasets/rgbd-dataset/download
  {
  std::string filename ("rgbd_dataset_freiburg1_desk/depth/1305031453.374112.png");

  cv::Mat depthImage; // 16bit unsigned
  depthImage = cv::imread(filename, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
  if (depthImage.empty()) {
    cerr << "loadDepthImage() error for loading: " << filename << endl;
    exit (0);
  }
  cout << "@ image loaded: " << filename << " : " << depthImage.cols << "x" << depthImage.rows << endl;

  depthImage.convertTo (depthImage, CV_32F);
  depthImage /= 5.; // TUM depth image, convert to mm unit

  zimg = depthImage;

  std::string rgbfilename ("rgbd_dataset_freiburg1_desk/rgb/1305031453.359684.png");
  cv::Mat image = cv::imread(rgbfilename, CV_LOAD_IMAGE_COLOR);
  if (image.empty()) {
          cerr << " -- RGB read error: " << filename << endl;
          exit (0);
  }
  rgb = image;
  }
}; // RGBZData

//
// ----------------------------------------------------------------------------------------------
//
struct UMM 
{
  void *operator new(size_t len) 
  {
    void *ptr;
    cudaMallocManaged(&ptr, len);
    cudaDeviceSynchronize();
    return ptr;
  }

  void operator delete(void *ptr) {
    cudaDeviceSynchronize();
    cudaFree(ptr);
  }
}; // UMM



template<typename T>
struct cuImage : UMM {
  T *data;
  int width, height;

  cuImage(int w, int h) { data=nullptr; alloc (w,h); }
  cuImage() { data = nullptr; }
  cuImage(const cv::Mat_<float>& m) { allocSet(m.rows, m.cols, m.data); }
  cuImage(const cv::Mat_<cv::Vec3b>& m) { allocSetBGR(m.rows, m.cols, m.data); }

  ~cuImage() { if (data) cudaFree (data); }

  void alloc (int h, int w) { 
    if (data) delete data;
    width=w, height=h; 
    cudaMallocManaged (&data, sizeof(float)*width*height);
  }

  void allocSet (int h, int w, void *img) {
    alloc(h,w);
    memcpy (data, img, sizeof(T)*width*height);
    //cudaMemcpy (data, img, sizeof(T)*width*height, cudaMemcpyHostToDevice);
    // since data is UM, we may use memcpy in the host side.
  }

  void allocSetBGR (int h, int w, void *img) { // use cuImage<uchar3> for rgb or bgr
    alloc(h,w);
    memcpy (data, img, sizeof(T)*width*height);
    //cudaMemcpy (data, img, sizeof(T)*width*height, cudaMemcpyHostToDevice);
    // since data is UM, we may use memcpy in the host side.
  }

  void zero() {
    memset(data, 0, sizeof(T)*width*height);
  }

  __device__ __host__ inline T& at(int y, int x) {
    return data[x + y*width];
  }

  __device__ __host__ inline T& operator()(int y, int x) {
    return data[x + y*width];
  }

  __device__ __host__ inline T& operator()(int y, int x) const {
    return data[x + y*width];
  }

  __device__ __host__ double bilinear(float v, float u) {
    int iu = u;
    double fu = u - iu;
    int iv = v;
    double fv = v - iv;
    double i1 = (1-fv) * this->at(iv, iu)   +  fv*this->at(iv+1, iu);
    double i2 = (1-fv) * this->at(iv, iu+1) +  fv*this->at(iv+1, iu+1);
    double r = (1-fu)*i1 + fu*i2;
    return r;
  }

  friend std::ostream& operator<< (std::ostream& os, const cuImage<T>& mat)
    {
    for (int r=0; r<mat.height; r++) {
      for (int c=0; c<mat.width; c++) 
        os << mat(r,c) << ' ';
      os << endl;
    }
    os << " --- " << endl;
    return os;
    }
};


template<typename T> 
struct cuMat : cuImage<T> {

        
  /***/
  cuMat<T>& operator=(cuMat<T> a) {
    if (this->width != a.width || this->height != a.height)
      this->alloc (a.width, a.height);
    // copy
    std::copy(a.data, a.data+a.width*a.height, this->data);
    //for (int r=0; r<this->height; r++) for (int c=0; c<this->width; c++) (*this)(r,c) = a(r,c);
    return (*this);
  }
  /***/

  /***
      cuMat<T>& operator=(const cuMat<T>& a)  // copy assignment
      {
      cerr << "-- copy operator= " << endl;
      if (this->width != a.width || this->height != a.height)
      this->alloc (a.width, a.height);
      // copy
      std::copy(a.data, a.data+a.width*a.height, this->data);
      //for (int r=0; r<this->height; r++) for (int c=0; c<this->width; c++) (*this)(r,c) = a(r,c);
      return (*this);
      }
  ***/

  /***
      cuMat<T>& operator=(cuMat<T>&& a) noexcept  // move assignment
      {
      cerr << "-- move operator= " << endl;
      if (this != &a) {
      if (this->data) cudaFree(this->data);
      this->data = a.data;
      a.data = nullptr;
      this->width = a.width;
      this->height = a.height;
      }
      return *this;
      }
  ***/

  cuMat<T> inv () {
    cv::Mat_<float> invmat = cv::Mat_<float> (this->width, this->height, this->data).inv();
    cuMat<T> i;
    i.set( invmat );
    return i;
  }

  void inverse (cuMat<T>& mat) {
    cv::Mat_<float> invmat = cv::Mat_<float> (this->width, this->height, mat.data).inv();
    (*this).set( invmat );
  }

  void set (cv::Mat_<float>& a) {
    if (this->width != a.cols || this->height != a.rows)
      this->alloc (a.cols, a.rows);
    // copy
    for (int r=0; r<this->height; r++) for (int c=0; c<this->width; c++) (*this)(r,c) = a(r,c);
  }

  void print (string s) {
    cout << s << endl;
    for (int r=0; r<this->height; r++) {
      for (int c=0; c<this->width; c++) 
	printf(" %f ", (*this)(r,c));
      printf("\n");
    }
    cout << " --- " << endl;
  }

  void eye() {
    for (int r=0; r<this->height; r++) 
      for (int c=0; c<this->width; c++) 
	(*this)(r,c) = r==c ? 1. : 0 ;
  }

  //friend cuMat<T> inverse(cuMat<T>);
};

/***
    cuMat<T> inverse(cuMat<T> m)
    {
    cv::Mat_<float> invmat = cv::Mat_<float> (m.width, m.height, m.data).inv();
    cuMat<T> i;
    i.set( invmat );
    return i;
    }
***/

struct Voxel  // 4*5 = 20 bytes
{
  float sdf;
  float weight;
  float color[3];
};


struct Camera {
  float fx, fy, cx, cy;
  float *K;
  cuMat<float> Hw2c, Hc2w; // from cam to world:   X_W = H(mat) * X_c

  //cv::Mat_<float> H; // 

  Camera() {
    K = nullptr; // c++11
    //alloc();
  }
  ~Camera() { 
    if (K) { cudaFree(K); }
  }

  void alloc() {
    if (K) { cudaFree(K); }
    cudaMallocManaged(&K, sizeof(float)*4);
    //cudaMallocManaged(&mat, sizeof(float)*16);
    //H = cv::Mat_<float>(4,4, mat);
    //printf ("@ Camera Created. %lu  %lu\n", (size_t)K, (size_t)H.data);
    Hw2c.alloc (4,4);
    Hc2w.alloc (4,4);
    cerr << "-- Hw2c allocated." << endl;
  }
  void setTUMCamera () {
    fx = 520, fy = 525, cx=319.5, cy = 239.5;
    K[0] = fx, K[1] = fy, K[2] = cx, K[3] = cy;
  }

  void setDefaultH()
  {
    float3 origin {-500, -500, 400}; // origin of tsdf w.r.t camera coordinate system
    Hw2c.eye();
    Hw2c(0,3) = origin.x;
    Hw2c(1,3) = origin.y;
    Hw2c(2,3) = origin.z;
    Hw2c.print("-- H_world2cam");

    Hc2w = Hw2c.inv();
    //Hc2w.inverse(Hw2c);
    //Hw2c.inverse(Hc2w);
    Hc2w.print("-- H_cam2world");
    Hw2c.print("-- H_world2cam");

    //cuMat<float> a;
    //a = Hw2c.inv();
    //a.print("-- aaaa H_c2w");
  }

  void setH(cv::Mat_<float> m)
  {
    //for (int r=0; r<4; r++) for (int c=0; c<4; c++) H(r,c) = m(r,c);
  }

  __device__ __host__ inline float3 getCameraCoord (float3& w)
  {
    float3 c;
    int i=0;
    c.x = Hw2c(i,0)*w.x + Hw2c(i,1)*w.y + Hw2c(i,2)*w.z + Hw2c(i,3);
    i=1;
    c.y = Hw2c(i,0)*w.x + Hw2c(i,1)*w.y + Hw2c(i,2)*w.z + Hw2c(i,3);
    i=2;
    c.z = Hw2c(i,0)*w.x + Hw2c(i,1)*w.y + Hw2c(i,2)*w.z + Hw2c(i,3);
    return c;
  }
};


#define VoxelDistanceUnit (4.f) // mili meter
#define TruncMargin (VoxelDistanceUnit*5.f) // sdf will be truncated with this threshold

struct TSDF 
{
  float3 origin; // the coord vector of tsdf origin w.r.t the 1st camera coordinate system
  // direction of TSDF space is the same as those of 1st camera
  float voxelUnit;

  Voxel *voxel;
  unsigned dimx, dimy, dimz;
  size_t size;
  float3 _maxcoord;

  TSDF(int dim=0) {
    size = 0;
    //init (dim==0? 700: dim);

    origin.x = -500;
    origin.y = -500;
    origin.z = 400;
    voxelUnit = VoxelDistanceUnit; // mili-meter
  }

  void init (int dim=500) {
    if (size != 0) cudaFree (voxel);
    dimx = dimy = dimz = dim;
    size = dimx*dimy*dimz;
    _maxcoord = float3 {dimx * VoxelDistanceUnit, dimy * VoxelDistanceUnit, dimz * VoxelDistanceUnit};

    printf ("@ TSDF size = %.2f giga bytes.\n", (double)size * sizeof(Voxel) / GB);
    printf ("@ sizeof (size_t) = %lu sizeof(Grid*)=%lu\n", sizeof (size_t), sizeof(Voxel*));
    //voxel = new Grid [s] ;
    cudaMallocManaged ((void**)&voxel, size*sizeof(Voxel));
    memset (voxel, 0, size*sizeof(Voxel)); // clear buffer

    printf ("@ Voxel allocated: %ux%ux%u = (%.1fx%.1fx%.1f)\n", dimx, dimy, dimz, _maxcoord.x, _maxcoord.y, _maxcoord.z);
  }

  ~TSDF () {
    if (voxel)
      cudaFree (voxel);
  }

  void setZero() {
    printf ("-- calling 3D kernel\n");
    dim3 block(1024,1,1);
    dim3 grid(32,32,32);

    kernel_setZero <<<grid,block>>> (*this);
    cudaCheckError();
    cudaDeviceSynchronize ();
  }

  void integrate(Camera* camera, cuImage<float>* pdepth, cuImage<uchar3>* pRGB)
  {
    printf("@@@ integrate a dpeth to TSDF @@@\n");
    float min=1E10, max = 0;
    cuImage<float>& depth = *pdepth;
    for (int i=0; i < depth.height; i++)
      for (int j=0; j < depth.width; j++) {
	if (depth.at(i,j) == 0) continue;
	if (depth.at(i,j) < min) min = depth.at(i,j);
	if (depth.at(i,j) > max) max = depth.at(i,j);
      }
    printf ("@ depth: min = %f  max = %fmm\n", min, max);
    printf ("@ now start kernel_integrate().\n");

    dim3 block(32,8,4);
    dim3 grid(divup(dimx, block.x), divup(dimy, block.y), divup(dimz, block.z));
    kernel_integrate<<<grid,block>>> (*this, depth, *camera);
    cudaCheckError();
    cudaDeviceSynchronize ();
  }

  void filesave (string filename)
  {
          typedef std::tuple<float, float, float, unsigned> dataElem;
          string fileext = filename.substr(filename.find_last_of(".")+1);
          if (fileext == "pcd") {
                  std::vector<dataElem> data;
                  unsigned dyz = dimy * dimz;
                  for (size_t i = 0 ; i < this->size; i++) {
                          if (voxel[i].weight > 0) {
                                float z  = VoxelDistanceUnit * (i / dyz);
                                unsigned xy = i % dyz;
                                float y = VoxelDistanceUnit * (xy / dimy);
                                float x = VoxelDistanceUnit * (xy % dimy);
                                unsigned color = 0xFFFF00 ; // yellow
                                auto c = std::make_tuple (x,y,z,color);
                                data.push_back (c);
                          }
                  }

                  FILE *fp = fopen (filename.c_str(), "w");
                  fprintf(fp, "# .PCD v.7 - Point Cloud Data file format\n");
                  fprintf(fp, "VERSION .7\n");
                  fprintf(fp, "FIELDS x y z rgb\n");
                  fprintf(fp, "SIZE 4 4 4 4\n");
                  fprintf(fp, "TYPE F F F F\n");
                  fprintf(fp, "COUNT 1 1 1 1\n");
                  fprintf(fp, "WIDTH %zu\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n", data.size());
                  fprintf(fp, "POINTS %zu\nDATA ascii\n", data.size());
                  for (auto i = 0; i < data.size(); i++) {
                          fprintf (fp, "%g  %g %g   %x\n", 
                            std::get<0>(data[i]),
                            std::get<1>(data[i]),
                            std::get<2>(data[i]),
                            std::get<3>(data[i])
                          );
                  }
                  fclose (fp);

          }
          else {
                  printf ("!!!  TSDF::filesave(%s) unknown file type.\n", filename.c_str());
          }
  }
}; // TSDF


__global__ void kernel_rayCast(TSDF& tsdf, Camera& cam, cuImage<float>& depth, cuImage<uchar4>& rgb)
{
        //printf
}


struct RENDERER
{
  // camera internals
  const int width = 640, height = 480;
  float fx , fy, u0, v0;
  cuImage<float> _depth;
  cuImage<uchar4> _rgb;

  void init() {
          _depth.alloc (width, height);
          _rgb.alloc (width, height);
  }

  void rayCasting (TSDF& tsdf, Camera& cam)
  {
    cout << "rayCasting for H_cam2world: " << endl <<  cam.Hc2w << endl;

    dim3 block(32,8,4);
    dim3 grid(divup(tsdf.dimx, block.x), divup(tsdf.dimy, block.y), divup(tsdf.dimz, block.z));

    kernel_rayCast <<<grid, block>>> (tsdf, cam, _depth, _rgb);
  }

};



// ----------------------------------------------------------------------------------------------

RGBZData DataReader;
TSDF *pTsdf;
RENDERER *pRenderer;
cuImage<float> *pDepth;
cuImage<uchar3> *pRGBImage;
Camera *pCamera;

__global__ void kernel_setZero (TSDF& tsdf) 
{
  size_t x = threadIdx.x + blockIdx.x * blockDim.x;
  size_t y = threadIdx.y + blockIdx.y * blockDim.y;
  size_t z = threadIdx.z + blockIdx.z * blockDim.z;

  size_t step = gridDim.x*blockDim.x*gridDim.y*blockDim.y*gridDim.z*blockDim.z;

  size_t offset =   z * gridDim.x*blockDim.x*gridDim.y*blockDim.y
    + y * gridDim.x*blockDim.x
    + x;

  while (offset < tsdf.size)
    {
      tsdf.voxel[offset].weight = 0;

      offset += step;
    }
}

__global__ void kernel_integrate (TSDF& tsdf, cuImage<float>& depth, Camera& camera)
{
  //  printf("@ kernel_integrate:: camera.K = {%f %f %f %f}\n", camera.K[0], camera.K[1], camera.K[2], camera.K[3]);
  //  printf("@-- depth: %dx%d, d[0] = %f\n", depth.width, depth.height, depth.at(depth.height/2,depth.width/2));

  size_t x = threadIdx.x + blockIdx.x * blockDim.x;
  size_t y = threadIdx.y + blockIdx.y * blockDim.y;
  size_t z = threadIdx.z + blockIdx.z * blockDim.z;


  if (x > tsdf.dimx || y > tsdf.dimy || z > tsdf.dimz) return;

  size_t offset = x + y * gridDim.x*blockDim.x +  z * gridDim.x*blockDim.x*gridDim.y*blockDim.y;

  // compute 3D coord vector w.r.t camera coord. system
      float3 p3w {(float)x*VoxelDistanceUnit,(float)y*VoxelDistanceUnit,(float)z*VoxelDistanceUnit}; // voxel coordinate in world
      float3 p3c = camera.getCameraCoord( p3w ); // coord in camera coord. system

  if (x == 0 && y == 0 && z == 0) {
          printf (">>> p3w = %f %f %f   p3c = %f %f %f\n", p3w.x, p3w.y, p3w.z, p3c.x, p3c.y, p3c.z);
  }

      if (p3c.z < 0) return; 

  // project to image pixel plane
      float u = camera.K[0] * p3c.x / p3c.z + camera.K[2];
      if (u < 0 || u > depth.width-2) return;

      float v = camera.K[1] * p3c.y / p3c.z + camera.K[3];
      if (v < 0 || v > depth.height-2) return;

      float dval = depth((int)v, (int)u); //.bilinear(v, u);
      if (dval <= 0.0 || dval > 3000.) return; // undefined depth value at (u,v)

      float diff = dval - p3c.z;

      if (diff < -TruncMargin || diff > TruncMargin) return;

      // integrate into TSDF
      float sdf = min (1.0f, diff / TruncMargin);
      float weight = 1;

      float weight_old = tsdf.voxel[offset].weight;
      tsdf.voxel[offset].weight += weight; // new updated weight

	  tsdf.voxel[offset].sdf = (weight_old * tsdf.voxel[offset].sdf + sdf * weight) / tsdf.voxel[offset].weight;
}

__global__ void kernel_test (Camera* camera)
{
  printf("-- kernel_test Camera* --\n");
  if (threadIdx.x == 0) {
    printf("-- kernel_test Camera* --\n");
    //printf("@ camera.K = {%f %f %f %f}\n", camera.K[0], camera.K[1], camera.K[2], camera.K[3]);
  }
}
__global__ void kernel_test (Camera& camera)
{
  if (threadIdx.x == 0) {
    //printf("-- kernel_test --\n");
    printf("@ kernel_test(Camera& camera):: camera.K = {%f %f %f %f}\n", camera.K[0], camera.K[1], camera.K[2], camera.K[3]);
  }
}

__global__ void kernel_test ()
{
  printf("@ kernel_test() for empty()\n");
}

int main() {

  cudaInfo();

  cudaMallocManaged (&pTsdf, sizeof(TSDF));
  TSDF &tsdf = *pTsdf;

  tsdf.init(512); // initial weight = 0
  //tsdf.setZero();

  cv::Mat_<float> I4 = cv::Mat_<float>::eye(4,4); 
  cudaMallocManaged(&pCamera, sizeof(Camera));
  pCamera->alloc();
  pCamera->setTUMCamera();
  pCamera->setDefaultH();

  printf("@@ kernel test for camera.\n");
  kernel_test<<<1,1>>> (*pCamera);
  cudaCheckError();
  cudaDeviceSynchronize ();

  DataReader.loadTUMSampleData();
  cv::Mat_<float> depthImage = DataReader.zimg;
  printf("@ depthImage = %f\n", depthImage(depthImage.rows/2, depthImage.cols/2));

  //cudaMallocManaged(&pDepth, sizeof(cuImage<float>));
  pDepth = new cuImage<float>;
  (*pDepth).allocSet(depthImage.rows, depthImage.cols, (void *)depthImage.data);

  pRGBImage = new cuImage<uchar3> (DataReader.rgb);
  // integration
  tsdf.integrate(pCamera, pDepth, pRGBImage);

  tsdf.filesave("output.pcd");

  // check result
  size_t count = 0;
  for (size_t i=0; i<tsdf.size; i++) {
          if (tsdf.voxel[i].weight > 0) count ++;
  }
  printf ("@@ tsdf.integrate() performed in %lu voxels.\n", count);

  // ray-casting
  cudaMallocManaged(&pRenderer, sizeof(RENDERER));
  pRenderer->init();

  pRenderer->rayCasting (*pTsdf, *pCamera);

  cv::waitKey ();

  cudaDeviceReset ();
  return 0;
}

// eof 
