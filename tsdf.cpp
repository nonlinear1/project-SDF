//
//
//
#include <vector>
#include <tuple>
#include <utility>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "/usr/local/cuda/samples/common/inc/helper_math.h"

using namespace std;


#define GB (1024*1024*1024)


//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {						\
		cudaError_t e=cudaGetLastError();		\
		if(e!=cudaSuccess) {											\
			printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
			exit(0);													\
		}																\
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

inline  unsigned divup(size_t a, size_t b) {
    unsigned r = (a+b-1)/b;
    printf ("@ divup(%zu %zu) = %u\n", a, b, r);
    return r;
}

inline dim3 getGrid(dim3 block, int dimx, int dimy=1, int dimz=1) {
	dim3 grid (divup(dimx, block.x), divup(dimy, block.y), divup(dimz, block.z));
	return grid;
}

// pre-declarations for function prototypes
//
struct TSDF;
template<typename T> struct cuImage;
struct Camera;

__global__ void kernel_integrate (TSDF& tsdf, cuImage<float>& depth, Camera& camera);
__global__ void kernel_setValue (TSDF& tsdf, float value); 
__global__ void kernel_setVoxel (TSDF& tsdf, float sdf, float weight); 
__global__ void kernel_test (cuImage<float>& depthImage);

__host__ __device__ inline float lerp (float x, float xmin, float xmax, float ymin, float ymax)
{
    float y = (ymax - ymin) * (x - xmin) / (xmax - xmin) + ymin;
    return y;
}

__host__ __device__ inline uchar4 lerp (float x, float xmin, float xmax, uchar4 rgba0, uchar4 rgba1)
{
    uchar4 y ;
    y.x = (uchar)lerp (x, xmin, xmax, rgba0.x, rgba1.x);
    y.y = (uchar)lerp (x, xmin, xmax, rgba0.y, rgba1.y);
    y.z = (uchar)lerp (x, xmin, xmax, rgba0.z, rgba1.z);
    y.w = 255;
    return y;
}


/////////////////////////////////////////////////////////////////////////////////////
const float ScaleCV2MM_TUM = 1./5.;
struct RGBZData {
	cv::Mat_<float> zimg;
	cv::Mat_<cv::Vec3b> rgb;

	static void writeTUM (cv::Mat_<float> cvd, string outfile) {
		cout << "@@ writeTUM" << endl;
		float maxd=0, mind=1E10;
		for (int r=0; r<cvd.rows; r++) {
			for (int c = 0; c < cvd.cols; c++) {
				if (cvd(r,c) > 1.) {
					//printf ("(%d %d:  %g) ", r, c, cvd(r,c));
					if (maxd < cvd(r,c)) maxd = cvd(r,c);
					if (mind > cvd(r,c)) mind = cvd(r,c);
				}
			}
		}
		printf ("\n\n@ writeTUM(): min, max = %g, %g\n", mind, maxd);
        cvd /= ScaleCV2MM_TUM;
        cv::Mat cvu16;
        cvd.convertTo (cvu16, CV_16U);
        string dimgFileName = outfile.empty() ? string("out-raycast.png") : outfile;
        cv::imwrite (dimgFileName, cvd);
        cout << "-- file saved : " << dimgFileName << endl;
	}
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
		depthImage *= ScaleCV2MM_TUM; // TUM depth image, convert to mm unit

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

	cuImage(int rows, int cols) { data=nullptr; alloc (rows, cols); }
	cuImage() { data = nullptr; width = 0; height = 0; }
	cuImage(const cv::Mat_<float>& m) { allocSet(m.rows, m.cols, m.data); }
	cuImage(const cv::Mat_<cv::Vec3b>& m) { allocSetBGR(m.rows, m.cols, m.data); }

	~cuImage() { if (data != nullptr) cudaFree (data); }

	cuImage<T>& operator=(cuImage<T>& a) {
		if (this->width != a.width || this->height != a.height) {
			this->dealloc();
			this->alloc (a.width, a.height);
		}
		for (int r=0; r<this->height; r++)
			for (int c=0; c<this->width; c++)
				(*this)(r,c) = a(r,c);
		return (*this);
	}

	void dealloc() {
		printf ("$ will be freed data =  %lu\n", (size_t) data);
		if (data != nullptr)
			cudaFree ( data );
	}
	
	void alloc (int h, int w) { 
		printf ("$ alloc(%d, %d)\n", h, w);
		width=w, height=h; 
		cudaMallocManaged (&data, sizeof(T)*width*height);
	}

	void allocSet (int h, int w, void *img) {
		//printf ("$ allocSet(%d, %d, %lu)\n", h, w, (size_t) img);
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
		if (y < 0 || y >= height || x < 0 || x >= width)
			printf ("\n\n illigal address cuImage(%d,%d)\n\n", y, x);
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
struct cuMat : public cuImage<T> {

	cuMat(int r, int c) : cuImage<T>(r,c) {}
	cuMat() : cuImage<T>() {}

    __host__ __device__ inline int rows() { return this->height; }
    __host__ __device__ inline int cols() { return this->width; }

	cuMat<T>& operator=(cuMat<T> a) {
		if (this->width != a.width || this->height != a.height) {
			this->dealloc();
			this->alloc (a.width, a.height);
		}
		for (int r=0; r<this->height; r++)
			for (int c=0; c<this->width; c++)
				(*this)(r,c) = a(r,c);
		return (*this);		
	}

	/****
	cuMat<T> inv () {
		cv::Mat_<T> invmat = cv::Mat_<T> (this->width, this->height, this->data).inv();
		cuMat<T> i;
		i.set( invmat );
		return i;
	}
	
	friend cuMat<T> inverse(cuMat<T>& m)
    {
		cv::Mat_<float> invmat = cv::Mat_<float> (m.width, m.height, m.data).inv();
		cuMat<T> i;
		i.set( invmat );
		return i;
    }
	****/
	
	void inverse (cuMat<T>& mat) {
		cv::Mat_<float> invmat = cv::Mat_<float> (this->width, this->height, mat.data).inv();
		(*this).set( invmat );
	}

	void set (cv::Mat_<float>& a) {
		cerr << "set: this = (" << this->rows() << ", " << this->cols() << ", " << this->data
			 << ") " << endl;
		if (this->width != a.cols || this->height != a.rows)
			this->alloc (a.cols, a.rows);
		// copy
		for (int r=0; r<this->height; r++) for (int c=0; c<this->width; c++) (*this)(r,c) = a(r,c);
	}

	__device__ __host__ void print (char *s=NULL) {
		if (s) printf("%s:\n", s);
		for (int r=0; r<this->height; r++) {
			for (int c=0; c<this->width; c++) 
				printf(" %g ", (*this)(r,c));
			printf("\n");
		}
		printf(" --- \n");
	}

	void eye(int n) {
		cout << "eye: " << this->rows() << "x" << this->cols() << endl;
		if (this->rows() != n || this->cols() != n)
            this->alloc (n,n);
		for (int r=0; r<this->height; r++) 
			for (int c=0; c<this->width; c++) 
				(*this)(r,c) = r==c ? 1. : 0 ;
	}

	/**
	friend cuMat<T> operator*(cuMat<T> a, cuMat<T> b) 
	{
        cout << "cuMat<T> operator*" << endl;
        cout << "a=" << a << endl << "b=" << b << endl;

		cuMat<T> mul (a.rows(), b.cols());
		for (int r=0; r < mul.rows(); r++)
			for (int c=0; c < mul.cols(); c++) {
				double sum = 0.;
				for (int k=0; k < a.cols(); k++)
					sum += a(r,k)*b(k,c);
				mul(r,c) = sum;
			}

		cout << "mul=" << mul << endl;
		return mul;
	}
	**/
	
	void setMul (cuMat<T>& a, cuMat<T>& b)
	{
        cout << "cuMat<T>::setMul()" << endl;
        cout << "a=" << a << endl << "b=" << b << endl;

		if (this->rows() != a.rows() || this->cols() != b.cols()) {
			printf ("\n\n\nmatrix multiplication size mis match!\n\n\n");
		}
		
		for (int r=0; r < this->rows(); r++)
			for (int c=0; c < this->cols(); c++) {
				double sum = 0.;
				for (int k=0; k < a.cols(); k++)
					sum += a(r,k)*b(k,c);
				this->at(r,c) = sum;
			}
		cout << "mul=" << (*this) << endl;
	}

};

struct Voxel  // 4*3 = 20 bytes
{
	float sdf;
	float weight;
	uchar4 rgba;
};


struct Camera {
	float fx, fy, cx, cy;
	float *K;
	cuMat<float> Hw2c, Hc2w; // from cam to world:   X_W = H(mat) * X_c
	cuMat<float> H20; // from current cam to reference camera
	//  uv = K[I|0] * X = K[I|0] H20^{-1} H20 X
	//  ! (H20 X) is the coordinate in the 0-th (reference) camera coordinate system
	//  ! Any point in 0-th camera coor. system will be projected to the current image through
	//    the camera K[I|0]*H20^{-1}

	//cv::Mat_<float> H; // 

	Camera() {
		K = nullptr; // c++11
		//alloc();
	}
	~Camera() { 
		if (K) { cudaFree(K); }
	}

	void alloc() {
		//if (K) { cudaFree(K); }
		cudaMallocManaged(&K, sizeof(float)*4);
		//cudaMallocManaged(&mat, sizeof(float)*16);
		//H = cv::Mat_<float>(4,4, mat);
		//printf ("@ Camera Created. %lu  %lu\n", (size_t)K, (size_t)H.data);
		Hw2c.alloc (4,4);
		Hc2w.alloc (4,4);
		H20.alloc (4,4);
		cerr << "-- Hw2c allocated." << endl;
	}
	void setTUMCamera () {
		fx = 520, fy = 525, cx=319.5, cy = 239.5;
		K[0] = fx, K[1] = fy, K[2] = cx, K[3] = cy;
	}

	void setDefaultH()
	{
		H20.eye(4);
		Hw2c.eye(4);
		Hc2w.inverse( Hw2c );
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


#define DepthMin (300.) // 300 mm meter, min depth for any camera
#define DepthMax (3000.) // 3000 mm meter, max depth for any camera

#define VoxelDistanceUnit (2.f) // mili meter
#define TruncMargin (VoxelDistanceUnit*5.f) // sdf will be truncated with this threshold

struct TSDF 
{
	cuMat<float> H02world; // transformation from reference camera to world (TSDF)
	float3 origin; // the coord vector of tsdf origin w.r.t the 1st camera coordinate system
	// direction of TSDF space is the same as those of 1st camera
	float voxelUnit;

	Voxel *voxel;
	unsigned dimx, dimy, dimz;
	size_t size;
	float3 _maxcoord;

	TSDF(int dim) { init (dim); }
	TSDF(int dx, int dy, int dz) { init(dx, dy, dz); }

	__device__ __host__ inline Voxel& at(size_t offset) {
		if (offset < size)
			return voxel[offset];
		else {
			printf ("\n\n -- wrong offset for voxel --\n\n");
			//exit (0);
		}
	}

	__host__ __device__ inline Voxel& at(int x, int y, int z) {
		size_t index = x + y * dimx + z * dimx * dimy;
		if (index < size)
			return voxel[index];
		else {
			printf ("\n\n -- wrong index for voxel (%d %d %d => %lu) dim(%u,%u,%u) --\n\n",
					x, y, z, index, dimx, dimy, dimz);
			//exit (0);
		}
	}
	
	__host__ __device__ inline Voxel& operator()(int x, int y, int z) {
		size_t index = x + y * dimx + z * dimx * dimy;
		return voxel[index];
	}

	__host__ __device__ float getSDF(float x, float y, float z) {
		unsigned ix = (unsigned) x;
		unsigned iy = (unsigned) y;
		unsigned iz = (unsigned) z;

		Voxel& vxl = this->operator()(ix, iy, iz);
		float sdf = vxl.sdf; // nearest neighbor
		return sdf;
	}

	__host__ __device__ uchar4 getColor(float x, float y, float z) {
		unsigned ix = (unsigned) x;
		unsigned iy = (unsigned) y;
		unsigned iz = (unsigned) z;

		Voxel& vxl = this->operator()(ix, iy, iz);
		uchar4 rgba = vxl.rgba; // nearest neighbor
		return rgba;
	}

	void init (int dx=500, int dy=300, int dz=2000, float z0=500, float distUnit = 4.) {
		if (dx <= 0 || dy <= 0 || dz <= 0) return;

		origin.x = -dx*VoxelDistanceUnit/2; // measured from the 0-th camera coordinate system
		origin.y = -dy*VoxelDistanceUnit/2;
		origin.z = z0;
		voxelUnit = VoxelDistanceUnit; // mili-meter

		cout << "H02world.eye(4)" << endl;
		H02world.eye(4);
		H02world (0,3) = -origin.x;
		H02world (1,3) = -origin.y;
		H02world (2,3) = -origin.z;
		cout << H02world << endl;

		dimx = dx, dimy = dy, dimz = dz;
		size = dimx*dimy*dimz;
		_maxcoord = float3 {dimx * VoxelDistanceUnit, dimy * VoxelDistanceUnit, dimz * VoxelDistanceUnit};

		printf ("@ TSDF size = %.2f giga bytes.\n", (double)size * sizeof(Voxel) / GB);
		printf ("@ sizeof (size_t) = %lu sizeof(Grid*)=%lu\n", sizeof (size_t), sizeof(Voxel*));
		//voxel = new Grid [s] ;
		cudaMallocManaged ((void**)&voxel, size*sizeof(Voxel));
		memset (voxel, 0, size*sizeof(Voxel)); 
		this->setUnexplored(); // the voxel space is filled with -1, Unexplored!

		printf ("@ Voxel allocated: %ux%ux%u = (%.1fx%.1fx%.1f)\n", dimx, dimy, dimz, _maxcoord.x, _maxcoord.y, _maxcoord.z);
	}

	~TSDF () {
		if (voxel)
			cudaFree (voxel);
	}

	void setUnexplored() {
		printf ("-- calling 3D kernel\n");
		dim3 block(1024,1,1);
		dim3 grid(divup(dimx, block.x), divup(dimy, block.y), divup(dimz, block.z));

		kernel_setVoxel <<<grid, block>>> (*this, -1./*sdf*/, 0./*weight*/);
		cudaCheckError();
		cudaDeviceSynchronize ();
	}

	void setZero() {
		printf ("-- calling 3D kernel\n");
		dim3 block(1024,1,1);
		dim3 grid(32,32,32);

		kernel_setValue <<<grid,block>>> (*this, 0.);
		cudaCheckError();
		cudaDeviceSynchronize ();
	}

	void integrate(Camera* camera, cuImage<float>* pdepth, cuImage<uchar3>* pRGB)
	{
		printf("@@@ integrate a dpeth to TSDF @@@\n");
		float min=1E10, max = 0;
		for (int i=0; i < pdepth->height; i++)
			for (int j=0; j < pdepth->width; j++) {
				if (pdepth->at(i,j) == 0) continue;
				if (pdepth->at(i,j) < min) min = pdepth->at(i,j);
				if (pdepth->at(i,j) > max) max = pdepth->at(i,j);
			}
		printf ("@ depth: min = %f  max = %fmm\n", min, max);
		if (10) {
			dim3 block (32,32);
			dim3 grid = getGrid (block, pdepth->width, pdepth->height);
			printf("-- gird: %d %d %d block: %d %d %d\n", 
				   grid.x, grid.y, grid.z, block.x, block.y, block.z);
			kernel_test<<<grid,block>>> (*pdepth);
			cudaCheckError();
			cudaDeviceSynchronize ();
			printf ("@ kernel_test for cuImage<float> depth finished.\n");
		}
		
		cout << "this->H02world: " << endl << this->H02world 
			 << "camera->H20:" << endl << camera->H20
			 << endl;
	   
		// camera->Hc2w = TSDF.H^{-1} * camera->H_pose
		//		camera->Hc2w = this->H02world * camera->H20;
		camera->Hc2w.setMul (this->H02world, camera->H20);
		cout << "-- computing inverse friend -- " << endl;
		camera->Hw2c.inverse( camera->Hc2w );
		cerr << "camera->Hc2w: " << endl << camera->Hc2w << endl;
		cerr << "camera->Hw2c: " << endl << camera->Hw2c << endl;
		
		printf ("@ now start kernel_integrate().\n");

		dim3 block(32,8,4);
		dim3 grid = getGrid (block, dimx, dimy, dimz); 
		kernel_integrate<<<grid,block>>> (*this, *pdepth, *camera);
		cudaCheckError();
		cudaDeviceSynchronize ();

		cerr << "@@@ integrate finished.\n" ;
	}

	void filesave (string filename)
	{
		typedef std::tuple<float, float, float, unsigned> dataElem;
		string fileext = filename.substr(filename.find_last_of(".")+1);
		if (fileext == "pcd") {
			std::vector<dataElem> data;
			for (size_t i = 0; i < this->dimx; i++)
				for (size_t j = 0; j < this->dimy; j++)
					for (size_t k = 0; k < this->dimz; k++) {
						Voxel &vxl = this->at(i,j,k);
						if (vxl.weight > 0 && (-.5 <= vxl.sdf && vxl.sdf <= .5) ) {
							float x = VoxelDistanceUnit * i;
							float y = VoxelDistanceUnit * j;
							float z = VoxelDistanceUnit * k;
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

			cerr << "@@ file saved to " << filename << " with " << data.size() << " elems." << endl;

		}
		else {
			printf ("!!!  TSDF::filesave(%s) unknown file type.\n", filename.c_str());
		}
	}
}; // TSDF


__global__ void kernel_rayCast(TSDF& tsdf, Camera& cam, cuImage<float>& _oz, cuImage<uchar4>& _obgra)
{
	unsigned u = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned v = threadIdx.y + blockIdx.y * blockDim.y;
	if (u >= _oz.width || v >= _oz.height) return;

	bool flag = false;
	
	float3 cray { (u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, 1.f };
	float3 ray = { 
		cam.Hc2w(0,0) * cray.x + cam.Hc2w(0,1) * cray.y + cam.Hc2w(0,2) * cray.z ,
		cam.Hc2w(1,0) * cray.x + cam.Hc2w(1,1) * cray.y + cam.Hc2w(1,2) * cray.z ,
		cam.Hc2w(2,0) * cray.x + cam.Hc2w(2,1) * cray.y + cam.Hc2w(2,2) * cray.z ,
	};
	float3 orig { cam.Hc2w(0,3), cam.Hc2w(1,3), cam.Hc2w(2,3) };

	if (u==320 && v==240) {
		printf ("@>> (%u,%u) o=(%g, %g, %g) r=(%g, %g, %g)\n", u, v,
				orig.x, orig.y, orig.z,
				ray.x, ray.y, ray.z);
	}
	// the voxels should be somewhere at p(z) = orig + z * ray
	bool isSurface = false;
	bool inBand  = false;
	uchar4 rgba, prev_rgba;
	rgba = make_uchar4(0,0,0,255);
	float sdf, prev_sdf;
	float z, zest = 0.;
	float step = VoxelDistanceUnit;
	for (z = DepthMin; z < DepthMax; z += step) {
		float3 p = orig + z * ray;

		// check if p is inside the volume
		//
		float ix = p.x / VoxelDistanceUnit;
		if (ix < 0 || ix >= tsdf.dimx) continue;
		float iy = p.y / VoxelDistanceUnit;
		if (iy < 0 || iy >= tsdf.dimy) continue;
		float iz = p.z / VoxelDistanceUnit;
		if (iz < 0 || iz >= tsdf.dimz) continue;

		Voxel &vxl = tsdf.at((int)ix, (int)iy, (int)iz);
		if (flag && u==320 && v==240) {
			printf ("uv:(%u,%u) z = %g, inBand= %d p=(%g, %g, %g) ip=(%d,%d,%d)  vxl = sdf:%g, w:%g\n", 
					u, v, z, inBand, 
					p.x, p.y, p.z, 
					(int) ix,
					(int) iy,
					(int) iz,
					vxl.sdf, vxl.weight);
		}
		// if p is inside the truncated sdf region
		if (inBand == false && (-1. < vxl.sdf && vxl.sdf < 1.)) {
			inBand = true;
			step = step / 2;
		}

		if (inBand) {
			// examine the sdf at p via interpolation
			sdf = tsdf.getSDF (ix, iy, iz);
			rgba = tsdf.getColor (ix, iy, iz);
			if (u==320 && v==240) {
				printf ("uv:(%u,%u) z = %g, inBand= %d p=(%g, %g, %g) sdf = %g\n", u, v, z, inBand, 
						p.x, p.y, p.z, sdf);
			}
			if (sdf < 0) {
				zest = lerp (0, sdf, prev_sdf, z, z - step);
				rgba = lerp (zest, z, z - step, rgba, prev_rgba);
				isSurface = true;
				break;
			}
			prev_sdf = sdf;
			prev_rgba = rgba;
		}
	}
	if (flag && u==320 && v==240) {
		printf ("uv:(%u,%u) z = %g, estimated Z = %g\n", zest); 
	}

	_oz(v, u) = zest;
	//_obgra (v, u) = rgba;
} // kernel_rayCast


struct RENDERER
{
	// camera internals
	static const int width = 640, height = 480;
	float fx , fy, u0, v0;
	cuImage<float> *_depth;
	cuImage<uchar4> *_rgb;
	cv::Mat_<float> _cvDepth;

	void init() {
		printf ("-- RENDERER::init()\n");
		cudaMallocManaged (&_depth, sizeof(cuImage<float>));
		_depth->alloc (height, width);

		cudaMallocManaged (&_rgb, sizeof(cuImage<uchar4>));
		_rgb->alloc (height, width);
	}

	void rayCasting (TSDF& tsdf, Camera& cam)
	{
		cout << "-- RENDERER::rayCasting() for H_cam2world (cam.Hc2w): " << endl 
			 <<  cam.Hc2w << endl;

		cout << "-- RENDERER::rayCasting() now calling kernel_rayCast<<<>>> --" << endl;

		dim3 block(32,32);
		dim3 grid(divup(width, block.x), divup(height, block.y));
    
		printf("-- gird: %d %d %d block: %d %d %d\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
		kernel_rayCast <<<grid, block>>> (tsdf, cam, *_depth, *_rgb);

		cudaCheckError();
		cudaDeviceSynchronize();

		cv::Mat_<float> cvd = cv::Mat_<float> (height, width, _depth->data);
		RGBZData::writeTUM (cvd, "out-raycast.png");
	}

};



// ----------------------------------------------------------------------------------------------

__global__ void kernel_setVoxel (TSDF& tsdf, float sdf, float weight) 
{
	size_t x = threadIdx.x + blockIdx.x * blockDim.x;
	size_t y = threadIdx.y + blockIdx.y * blockDim.y;
	size_t z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x >= tsdf.dimx || y >= tsdf.dimy || z >= tsdf.dimz) return;

	tsdf.at(x,y,z).sdf = sdf;
	tsdf.at(x,y,z).weight = weight;
}

__global__ void kernel_setValue (TSDF& tsdf, float value) 
{
	size_t x = threadIdx.x + blockIdx.x * blockDim.x;
	size_t y = threadIdx.y + blockIdx.y * blockDim.y;
	size_t z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x >= tsdf.dimx || y >= tsdf.dimy || z >= tsdf.dimz) return;

	tsdf.at(x,y,z).weight = value;
}

__global__ void kernel_integrate (TSDF& tsdf, cuImage<float>& depth, Camera& camera)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x >= tsdf.dimx || y >= tsdf.dimy || z >= tsdf.dimz) return;

	bool flag = true && (x==0 && y==0 && z==0) ;

	//return;
	if (flag) {
        printf("@ kernel_integrate:: camera.K = {%f %f %f %f}\n",
			   camera.K[0], camera.K[1], camera.K[2], camera.K[3]);
        printf("@-- depth: %dx%d, d[0] = %f\n",
			   depth.width, depth.height, depth.at(depth.height/2,depth.width/2));
		printf ("camera: Hw2c\n");
		camera.Hw2c.print();
	}

	//return;
	// compute 3D coord vector w.r.t camera coord. system
	// voxel coordinate in world
	float3 p3w {(float)x*VoxelDistanceUnit,(float)y*VoxelDistanceUnit,(float)z*VoxelDistanceUnit};
	// coord in camera coord. system
	float3 p3c = camera.getCameraCoord( p3w ); 

	if (flag) {
		printf (">>> p3w = %f %f %f   p3c = %f %f %f\n", p3w.x, p3w.y, p3w.z, p3c.x, p3c.y, p3c.z);
		printf (" K= (%g %g %g %g)\n", camera.K[0], camera.K[1], camera.K[2], camera.K[3]);
	}

	if (p3c.z < 0) return; 

	// project to image pixel plane
	float u = camera.K[0] * p3c.x / p3c.z + camera.K[2];
	if (u < 0 || u > depth.width-2) return;

	float v = camera.K[1] * p3c.y / p3c.z + camera.K[3];
	if (v < 0 || v > depth.height-2) return;

	//return;
	float dval = depth.at((int)v, (int)u); //.bilinear(v, u);
	if (dval <= DepthMin || dval > DepthMax) return; // unused depth value at (u,v)

	float diff = dval - p3c.z;

	if (diff < -TruncMargin) return;

	// integrate into TSDF
	float sdf = min (1.0f, diff / TruncMargin); // sdf==1. means that voxel is empty space
	float weight = 1;

	//return;
	Voxel& vxl = tsdf.at(x, y, z);
	float weight_old = vxl.weight;
	vxl.weight += weight; // new updated weight
	vxl.sdf = (weight_old * vxl.sdf + weight * sdf) / vxl.weight ;

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

__global__ void kernel_test (cuImage<float>& depthImage)
{
	size_t x = threadIdx.x + blockIdx.x * blockDim.x;
	size_t y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= depthImage.width || y >= depthImage.height) return;

	float val = depthImage (y, x); // test

	val += 1; // any operation
}

__global__ void kernel_test ()
{
	printf("@ kernel_test() for empty()\n");
}

// ----------------------------------------------------------------------------------------------
//
RGBZData DataReader;
TSDF *pTsdf;
RENDERER *pRenderer;
cuImage<float> *pDepth;
cuImage<uchar3> *pRGBImage;
Camera *pCamera;

// ----------------------------------------------------------------------------------------------
//
int main() {

	cudaInfo();

	cudaMallocManaged (&pTsdf, sizeof(TSDF));
	TSDF &tsdf = *pTsdf;

	tsdf.init(640, 480, 2500, 700); // initial weight = 0
	//tsdf.init(50, 30, 2); // initial weight = 0

	cv::Mat_<float> I4 = cv::Mat_<float>::eye(4,4); 
	cudaMallocManaged(&pCamera, sizeof(Camera));
	memset (pCamera, 0, sizeof(Camera));
			
	pCamera->alloc();
	pCamera->setTUMCamera();
	pCamera->setDefaultH(); // I

	printf("@@ kernel test for camera.\n");
	kernel_test<<<1,1>>> (*pCamera);
	cudaCheckError();
	cudaDeviceSynchronize ();

	DataReader.loadTUMSampleData();
	cv::Mat_<float> depthImage = DataReader.zimg;
	printf("@ depthImage(%d, %d) = %f\n", depthImage.rows/2, depthImage.cols/2,
		   depthImage(depthImage.rows/2, depthImage.cols/2));

	printf("@ cuImage<float> *pDepth allocation.\n");
	cudaMallocManaged(&pDepth, sizeof(cuImage<float>));
	memset (pDepth, 0, sizeof(cuImage<float>));
	//pDepth = new cuImage<float>;
	pDepth->allocSet(depthImage.rows, depthImage.cols, (void *)depthImage.data);

	//pRGBImage = new cuImage<uchar3> (DataReader.rgb);

	if (0) {
		printf ("@@@ Before integration @@@\n");
		for (auto i=0; i < tsdf.size; i++) {
			printf ("voxel[%d] = (%g %g)\n", i, tsdf.voxel[i].sdf, tsdf.voxel[i].weight);
		}
	}

	// integration
	tsdf.integrate(pCamera, pDepth, pRGBImage);
	tsdf.filesave("output.pcd");

	if (0) {
		printf ("@@@ After integration @@@\n");
		for (auto i=0; i < tsdf.size; i++) {
			printf ("voxel[%d] = (%g %g)\n", i, tsdf.voxel[i].sdf, tsdf.voxel[i].weight);
		}
	}

	if (10)
		{
		// ray-casting
			cudaMallocManaged(&pRenderer, sizeof(RENDERER));
			pRenderer->init();
			pRenderer->rayCasting (*pTsdf, *pCamera);
		}

	cv::waitKey ();

	cudaDeviceReset ();
	return 0;
}

// eof 
