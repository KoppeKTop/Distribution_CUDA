// Новая версия
// Что будет делать?
// В общем массиве на ГПУ будут храниться координаты сфер структуры
// В едином битовом массиве будут храниться уже отмеченные точки
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include "Indexer.h"
#include "Coord.h"
#include "cuda_helper.h"
#include "CalcPlan.h"
#include <cuda_runtime.h>
#include "Rect.h"

using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using std::ifstream;
using std::ios;
using std::min;

#define SQR(X) (X)*(X)
#define BIT_IN_INT 32
#define THREADS_IN_HELPERS 256

typedef vector<dCoord> SphereVec;

time_t time_from_start()
{
    static time_t start = time(NULL);
    return time(NULL) - start;
}

ostream& operator<<(ostream& out, dim3& x ) 
{
    out << "{x = " << x.x << ", y = " << x.y << ", z = " << x.z << "}";
    return out;
}

//Round a / b to nearest higher integer value
int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Align a to nearest higher multiple of b
int iAlignUp(int a, int b)
{
    return (a % b != 0) ?  (a - a % b + b) : a;
}

__host__ __device__ bool is_overlapped(const float4 sph1, const float4 sph2)
{
    float r_sum = SQR((float)(sph1.w + sph2.w));
    float r = SQR((float)(sph1.x - sph2.x)) + 
    SQR((float)(sph1.y - sph2.y)) + SQR((float)(sph1.z - sph2.z));
    return ((r - r_sum) < (float)(1e-4));
}

bool is_overlapped(const dCoord & sph1, const dCoord & sph2)
{
    float r_sum = SQR(float(sph1[3] + sph2[3]));
    float r = SQR((float)(sph1[0] - sph2[0])) + 
    SQR(float(sph1[1] - sph2[1])) + SQR(float(sph1[2] - sph2[2]));
    return ((r - r_sum) < float(1e-4));
}

__host__ __device__ int NumberOfSetBits(int i)
{
    i = i - ((i >> 1) & 0x55555555);
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
    return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

__global__ void overlap_kernel(float4 * spheres, float4 test_sph, int * result)
{
	if (is_overlapped(spheres[threadIdx.x], test_sph)) {
		result[threadIdx.x] = 1;
	}
}

// blockDim.x == 32!!!
// blockDim.y == 8 (or 4 or 16...)
// gridDim = (fld_size.y, 1)
// need shared mem == blockDim.y * 2 * sizeof(int)
// ont block computes 1 line in all layers
__global__ void get_overlapping_field(int * fld, float4 * spheres, int spheres_cnt, 
float radius, int z_cnt, float cell_len, int ints_per_line, int ints_per_layer
)
{
	const int bit_cnt = threadIdx.x;
	int z = 0;
	float4 curr_pnt;
    curr_pnt.y = (blockIdx.x + 0.5f) * cell_len;
    curr_pnt.z = (z + 0.5f) * cell_len;
    curr_pnt.w = radius;
	extern __shared__ int res []; // last element – mutex 

	if (threadIdx.x == 0)
	{
		res[threadIdx.y] = 0;
		if ( threadIdx.y == 0 )
			res[blockDim.y] = 0;
	}
	__syncthreads();
	for (;z < z_cnt; z++, curr_pnt.z += cell_len)
	{
		for (int int_cnt = threadIdx.y; int_cnt < ints_per_line; int_cnt += blockDim.y)
		{
		curr_pnt.x = (bit_cnt + int_cnt * 32 + 0.5f) * cell_len;
		bool overlapped = false;
		for (int sphIdx = 0; sphIdx < spheres_cnt; ++sphIdx)
		{
			if (is_overlapped(curr_pnt, spheres[sphIdx]))
			{
				overlapped = true;
				break;
			}
		}
		if (overlapped) {
			continue;
		}

		atomicOr(&res[threadIdx.y], 1 << (31 - threadIdx.x));
		bool mutex_get = (atomicCAS(&res[blockDim.y + threadIdx.y], 0, 1) == 0); 
		// if last element was == 0, then 
		// it set 1 and mutex for that integer get
		__syncthreads();
		if (mutex_get) {
			fld[int_cnt + blockIdx.x * ints_per_line + z * ints_per_layer] = res[threadIdx.y];
			res[threadIdx.y] = 0;
			//release mutex
			res[blockDim.y + threadIdx.y] = 0;
		}
		__syncthreads();
		}
	}
}

__global__ void xor_fields(int * fld1, int * fld2, int * dest, int cnt)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < cnt)
		dest[idx] = fld1[idx] ^ fld2[idx];
}

__global__ void or_fields(int * fld1, int * fld2, int * dest, int cnt)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < cnt)
		dest[idx] = fld1[idx] | fld2[idx];
}

struct Map
{
	/* data */
	short * x;
	short * y;
	short * z;
	int cnt;
};

// каждый поток будет двигать один int в соответствии с картой
// каждый блок будет двигать линию 
// размерность блока: ints_per_line, 1, 1
// количество блоков: fld_size.y, fld_size.z
// нужно (blockDim.x+1) * sizeof(int) байт shared памяти
__global__ void apply_map(int * centers_fld, int * result_fld, Map map,
int ints_per_line, int ints_per_layer)
{
	const int x = threadIdx.x;
	const int y = blockIdx.x;
	const int z = blockIdx.y;

	int new_x, new_y, new_z;

	const int templ = centers_fld[x + y * ints_per_line + z * ints_per_layer];

	int prev_shift = 0;
	// __shared__ int shifted_line [];
	// shifted_line[x] = templ;
	// if (x == 0)
	// {
	// 	shifted_line[blockDim.x] = 0;
	// }
	// __syncthreads();

	if (templ == 0)
	{
		return; // nothing to move
	}

	int bit_shift = 0;
	int int_shift = 0;
	int first_int = 0;
	int second_int = 0;

	for (int curr_map_idx = 0; curr_map_idx < map.cnt; curr_map_idx++)
	{
		int shift = map.x[curr_map_idx];
		bool neg_shift = false;
		if (shift < 0)
		{
			neg_shift = true;
			shift = -shift;
		}
		int_shift = (int) shift / 32;
		bit_shift = shift - int_shift * 32;
		if (bit_shift != prev_shift)
		{
			if (neg_shift)
			{
				// shift right
				first_int = templ << bit_shift;
				second_int = templ >> (32 - bit_shift);
			} else if (int_shift != 0) {
				first_int = templ >> bit_shift;
				second_int = templ << (32 - bit_shift);
			} else {
				first_int = templ;
				second_int = 0;
			}
		}
		if (neg_shift)
		{
			new_x = x - int_shift;
			new_y = y + map.y[curr_map_idx];
			new_z = z + map.z[curr_map_idx];
			if (0 <= new_x && 0 <= new_y && new_y < gridDim.x && 
				0 <= new_z && new_z < gridDim.y)
			{
				atomicOr(result_fld + new_x + new_y * ints_per_line + new_z * ints_per_layer, first_int);
				new_x -= 1;
				if (0 <= new_x) {
					atomicOr(result_fld + new_x + new_y * ints_per_line + new_z * ints_per_layer, second_int);
				}
			}
		} else {
			new_x = x + int_shift;
			new_y = y + map.y[curr_map_idx];
			new_z = z + map.z[curr_map_idx];
			if ( new_x < ints_per_line && 0 <= new_y && new_y < gridDim.x && 0 <= new_z && new_z < gridDim.y)
			{
				atomicOr(result_fld + new_x + new_y * ints_per_line + new_z * ints_per_layer, first_int);
				new_x += 1;
				if (new_x < ints_per_line) {
					atomicOr(result_fld + new_x + new_y * ints_per_line + new_z * ints_per_layer, second_int);
				}
			}
		}
	}
}

// считаем количество взведённых бит на поле
// размерность блока: ints_per_line, 1, 1
// количество блоков: fld_size.y, fld_size.z
__global__ void fld_cnt(int * fld, dim3 start_point, dim3 stop_point, int * result)
{
	const int start_x = threadIdx.x * 32;
	const int stop_x = start_x + 31;
	
	__shared__ int bits_in_line;
	if (threadIdx.x == 0)
	{
		bits_in_line = 0;
	}
	__syncthreads();

	if (stop_x < start_point.x || blockIdx.x < start_point.y || blockIdx.y < start_point.z || 
		start_x > stop_point.x || blockIdx.x > stop_point.y || blockIdx.y > stop_point.z )
	{
		return;
	}

	int current_val = fld[threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x];
	if (current_val == 0)
	{
		return;
	}

	if (start_x < start_point.x && start_point.x <= stop_x)
	{
		current_val &= 0xFFFFFFFF << (start_point.x - start_x);

	} else if (start_x <= stop_point.x && stop_point.x < stop_x)
	{
		current_val &= 0xFFFFFFFF >> (stop_x - stop_point.x);
	}

	atomicAdd(&bits_in_line, NumberOfSetBits(current_val));
	__syncthreads();
	atomicAdd(result, bits_in_line);
}

bool compare_shifts(const iCoord & first, const iCoord & second)
{
	return (first[0] % 32) < (second[0] % 32);
}

// generates map and stores it in GPU
// for maximum performance map must be sorted (by bit shifts)
Map generate_map(double radius, double cell_len)
// radius – радиус сферы
// a – сторона куба (квадрата)
{
    CoordVec * map = NULL;
    
    dCoord centre;
    int divCnt = ceil(2.0 * radius / cell_len);
    double centreCoord = divCnt / 2.0;
    for (int d = 0; d < dCoord::GetDefDims()-1; ++d) {
        centre[d] = centreCoord;
    }
    centre[dCoord::GetDefDims()-1] = divCnt/2.0;

    map = new CoordVec;
    vector<int> sz(3, divCnt);
    Indexer indx(sz);
    dCoord curr_coord;
    iCoord curr_icoord;
    while (!indx.is_last()) {
        vector<int> curr_vec = indx.curr();
        for (int d = 0; d < curr_vec.size(); ++d) {
            curr_coord[d] = curr_vec[d] + 0.5;
            curr_icoord[d] = curr_vec[d] - divCnt/2;
        }
        if (is_overlapped(centre, curr_coord)) {
            map->push_back(curr_icoord);
        }
        indx.next();
    }
    std::sort(map->begin(), map->end(), compare_shifts);
    Map result;
    int gpu_arr_sz = map->size() * sizeof(short);
    cudaSafeCall(cudaMalloc(&(result.x), gpu_arr_sz));
    cudaSafeCall(cudaMalloc(&(result.y), gpu_arr_sz));
    cudaSafeCall(cudaMalloc(&(result.z), gpu_arr_sz));

    short * x = new short[map->size()];
    short * y = new short[map->size()];
    short * z = new short[map->size()];
    for (int idx = 0; idx < map->size(); ++idx)
    {
    	x[idx] = map[0][idx][0];
    	y[idx] = map[0][idx][1];
    	z[idx] = map[0][idx][2];
    }
    cudaSafeCall(cudaMemcpy(result.x, x, gpu_arr_sz, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(result.y, y, gpu_arr_sz, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(result.z, z, gpu_arr_sz, cudaMemcpyHostToDevice));
    result.cnt = map->size();

    cout << map->size() << " points in map\n";

    delete [] x;
    delete [] y;
    delete [] z;
    delete map;
    return result;
}

int iteration(float4 * d_spheres, int spheres_cnt, int * d_fld, int * d_centers_fld, 
				const dim3 & fld_size, double radius, double cell_len, int * result_fld,
				dim3 start_point, dim3 stop_point)
{
	int ints_per_line = fld_size.x/BIT_IN_INT;
	int ints_per_layer = ints_per_line * fld_size.y;
	int total_bytes = fld_size.x * fld_size.y * fld_size.z / 8;
	int total_ints = ints_per_layer * fld_size.z;
	Map iter_map = generate_map(radius, cell_len);

	dim3 dim_block_over(BIT_IN_INT, 8, 1);
	dim3 dim_grid_over(fld_size.y, 1);
	dim3 dim_block_help(THREADS_IN_HELPERS, 1, 1);
	dim3 dim_grid_help((total_ints % dim_block_help.x == 0) ? (total_ints / dim_block_help.x) : (total_ints / dim_block_help.x + 1), 1);
	dim3 dim_block_map(ints_per_line, 1, 1);
	dim3 dim_grid_map(fld_size.y, fld_size.z);

    int * d_cells_cnt;
    int result;


    cout << dim_grid_over << endl;
    cout << dim_block_over << endl;
    cudaSafeCall(cudaMalloc(&d_cells_cnt, sizeof(int)));
    cudaSafeCall(cudaMemset(d_cells_cnt, 0, sizeof(int)));

	cudaSafeCall(cudaMemset(d_fld, 0, total_bytes));
        cout << time_from_start() << " overlapping... ";

	get_overlapping_field <<<dim_grid_over, dim_block_over, dim_block_over.y * 2 * sizeof(int)>>>
	(d_fld, d_spheres, spheres_cnt, radius, fld_size.z, cell_len, ints_per_line, ints_per_layer
	);
	cudaSafeCall(cudaDeviceSynchronize());
	cout << time_from_start() << ": " << " Done\n";
	xor_fields <<<dim_grid_help, dim_block_help>>> (d_fld, d_centers_fld, d_fld, total_ints);
	cudaSafeCall(cudaDeviceSynchronize());
	cout << time_from_start() << ": start apply map... ";
	apply_map <<<dim_grid_map, dim_block_map>>> (d_fld, result_fld, iter_map, ints_per_line, ints_per_layer);
	cudaSafeCall(cudaDeviceSynchronize());
	cout << time_from_start() << ": Done\n";
	or_fields <<<dim_grid_help, dim_block_help>>> (d_fld, d_centers_fld, d_centers_fld, total_ints);
	cudaSafeCall(cudaDeviceSynchronize());
	cout << time_from_start() << ": counting... " ;
	fld_cnt <<<dim_grid_map, dim_block_map>>> (result_fld, start_point, stop_point, d_cells_cnt);
	cudaSafeCall(cudaDeviceSynchronize());
	cout << time_from_start() << ": Done\n";

	cudaSafeCall(cudaMemcpy(&result, d_cells_cnt, sizeof(int), cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaFree(d_cells_cnt));
	cudaSafeCall(cudaFree(iter_map.x));
	cudaSafeCall(cudaFree(iter_map.y));
	cudaSafeCall(cudaFree(iter_map.z));

	return result;
}

double getVolume(const SphereVec & h_spheres, const Rect & box, double poreSize, double sq_len)
{
    static dCoord zero_pnt;
    static iCoord fld_size;
    static dim3 fld_dim;
    static dCoord scale;
    int divCnt = ceil(poreSize/sq_len);
    
    static size_t prev_cells_cnt = 0;
    static int * result_fld = NULL;
    static int * centers_fld = NULL;
    static int * curr_centers_fld = NULL;
    static float4 * d_spheres = NULL;
    static dim3 start_point, stop_point;
    
    double radius = poreSize/2.0;
    printf("Division count = %d\n", divCnt);

    if (result_fld == NULL) {
        // first launch
        size_t total_bits = 1;
        for (int dim = 0; dim < iCoord::GetDefDims(); ++dim) {
            fld_size[dim] = (int)((box.maxCoord[dim] - box.minCoord[dim]) / sq_len + 1);
            zero_pnt[dim] = box.minCoord[dim];
            scale[dim] = sq_len;
        }
        fld_size[0] = iAlignUp(fld_size[0], 32);

        fld_dim.x = fld_size[0];
        fld_dim.y = fld_size[1];
        fld_dim.z = fld_size[2];

        for (int dim = 0; dim < iCoord::GetDefDims(); ++dim) {
            total_bits *= fld_size[dim];
        }

        int ignore = (int) poreSize/sq_len/2.0;
        start_point.x = ignore;
        start_point.y = ignore;
        start_point.z = ignore;

        stop_point.x = fld_dim.x - ignore;
        stop_point.y = fld_dim.y - ignore;
        stop_point.z = fld_dim.z - ignore;

        scale[iCoord::GetDefDims()] = 1; // scale of radius
        size_t total_bytes = (size_t)total_bits/8;
	cout << "Need memory: " << total_bytes * 3 << endl;
        cudaSafeCall(cudaMalloc(&result_fld, total_bytes));
        cudaSafeCall(cudaMemset(result_fld, 0, total_bytes));
        cudaSafeCall(cudaMalloc(&centers_fld, total_bytes));
        cudaSafeCall(cudaMemset(centers_fld, 0, total_bytes));
        cudaSafeCall(cudaMalloc(&curr_centers_fld, total_bytes));
        cudaSafeCall(cudaMemset(curr_centers_fld, 0, total_bytes));

        // send to GPU
        float4 * h_spheres_floats = new float4[h_spheres.size()];

        for (int i = 0; i < h_spheres.size(); ++i)
        {
        	h_spheres_floats[i] = make_float4(h_spheres[i][0], h_spheres[i][1], h_spheres[i][2], h_spheres[i][3]);
        }
        cudaSafeCall(cudaMalloc(&d_spheres, h_spheres.size() * sizeof(float4)));
        cudaSafeCall(cudaMemcpy(d_spheres, h_spheres_floats, h_spheres.size() * sizeof(float4), 
        			cudaMemcpyHostToDevice));
        delete [] h_spheres_floats;
    }
    
    int cells_cnt = iteration(d_spheres, h_spheres.size(), curr_centers_fld, centers_fld, 
								fld_dim, radius, sq_len, result_fld, 
								start_point, stop_point);

    double one_cell_vol = pow(sq_len, iCoord::GetDefDims());

    int added_cells_cnt = cells_cnt - prev_cells_cnt;
    prev_cells_cnt = cells_cnt;
    return added_cells_cnt * one_cell_vol;
}

vector<double> getDistribution(const SphereVec & spheres, double minPores, double maxPores, double h, int divisions,
                                double field_size)
{
    Rect box;
    for (int dim = 0; dim < iCoord::GetDefDims(); ++dim) {
        box.minCoord[dim] = 0;
        box.maxCoord[dim] = field_size;
    }
    
    vector<double> result;
    double sq_len = min(h, minPores/divisions);
    
    vector<double> poreSizes;
    for (double poreSize = minPores; poreSize <= maxPores; poreSize += h) {
        poreSizes.push_back(poreSize);
    }
    for (int poreSizeIdx = poreSizes.size()-1; poreSizeIdx >= 0; --poreSizeIdx) {
        cout << time_from_start() << ": For pore size " << poreSizes[poreSizeIdx] << endl;
        result.push_back(getVolume(spheres, box, poreSizes[poreSizeIdx], sq_len));

        cout << time_from_start() << ": Result: " << result.back() << endl;
    }
    return result;
}

bool exists(const char *fname)
{
    if( access( fname, F_OK ) != -1 ) {
        return true;
    } else {
        return false;
    }
}

template <typename T>
void load_coords(const char * filename, SphereVec * v)
{
    if (filename == NULL || v == NULL) {
        fprintf(stderr, "Load coords error! Wrong args\n");
        exit(10);
    }
    
    FILE *ifile;
    T *buffer;
    unsigned long fileLen;
    int type_size = sizeof(T);
    
    ifile = fopen(filename, "rb");
    if (!ifile) {
        fprintf(stderr, "No such file: %s\n", filename);
        exit(11);
    }
    
    fseek(ifile, 0, SEEK_END);
    fileLen=ftell(ifile);
    fseek(ifile, 0, SEEK_SET);
    
    if (fileLen % type_size != 0 || fileLen/type_size % dCoord::GetDefDims() != 0) {
        fprintf(stderr, "Wrong file: %s\n", filename);
        exit(12);
    }
    
    buffer=(T *)malloc(dCoord::GetDefDims() * type_size);
    if (!buffer)
    {
        fprintf(stderr, "Memory error!");
        fclose(ifile);
        exit(13);
    }
    
    v->clear();
    while (fread(buffer, dCoord::GetDefDims(), type_size, ifile)) {
        dCoord curr;
        for (int i = 0; i < dCoord::GetDefDims(); ++i) {
            curr[i] = (double)buffer[i];
        }
        v->push_back(curr);
    }
    
    fclose(ifile);
    
    cout << v->size() << " points added\n";
}

int main(int argc, char ** argv)
{
    iCoord::SetDefDims(3);
    dCoord::SetDefDims(iCoord::GetDefDims()+1);
    Coord<size_t>::SetDefDims(iCoord::GetDefDims());
    SphereVec v;
    TCalcPlan plan(argc, argv);
    int divisions = plan.divisions;
    cudaSafeCall(cudaSetDevice(plan.gpu));
    time_from_start();
    
    if (plan.is_float) {
        load_coords<float>(plan.filename, &v);
    } else {
        load_coords<double>(plan.filename, &v);
    }
    
    vector<double> distr = getDistribution(v, 1.0, 50.0, 1.0, divisions, plan.field_size);
    for (vector<double>::reverse_iterator curr_d = distr.rbegin(); curr_d != distr.rend(); ++curr_d) {
        cout << *curr_d << " ";
    }
    cout << endl;
    return 0;
}

