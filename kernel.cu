
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Новая версия
// Что будет делать?
// В общем массиве на ГПУ будут храниться координаты сфер структуры
// В едином битовом массиве будут храниться уже отмеченные точки
#include <stdio.h>
//#include <io.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <algorithm>
#include "Indexer.h"
#include "Coord.h"
#include "cuda_helper.h"
#include "CalcPlan.h"
#include <cuda_runtime.h>
#include "Rect.h"
#include <fstream>

#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/count.h>

using thrust::device_ptr;
using thrust::transform_reduce;
using thrust::transform;
using thrust::counting_iterator;

using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::ios;
using std::min;

#define SQR(X) (X)*(X)
#define PI 3.14159265
#define BIT_IN_INT 32
#define THREADS_IN_HELPERS 256
#define _SLOW_TEST
#undef _SLOW_TEST

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

struct bits_cntr
{
	__host__ __device__ 
		int operator()(int i)
	{
		i = i - ((i >> 1) & 0x55555555);
		i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
		return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
	}
};

// count number of settet bits on GPU array
int fld_bit_cnt(int * fld, size_t elements_cnt)
{
	device_ptr<int> dev_ptr(fld);
	return transform_reduce(dev_ptr, dev_ptr + elements_cnt, bits_cntr(), 0, thrust::plus<int>());
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
// one block computes 1 line in all layers
// maximum - 5.5K shared mem (up to 12288 in each direction) 
#define MAX_INTS_PER_LINE 384

__global__ void get_overlapping_field(int * fld, float4 * spheres, int spheres_cnt, 
float radius, int z_cnt, float cell_len, int ints_per_line, int ints_per_layer, 
float map_shift
)
{
    const int bit_idx = threadIdx.x;
	const float fld_sz = cell_len * z_cnt;
    int z = 0;
    int jumps = ints_per_line/blockDim.y;
    if (jumps * blockDim.y < ints_per_line)
        jumps++;
    float4 curr_pnt;
    curr_pnt.y = (blockIdx.x + map_shift) * cell_len;
    curr_pnt.z = (z + map_shift) * cell_len;
    curr_pnt.w = radius;
    __shared__ int res[MAX_INTS_PER_LINE]; // temporary results
    __shared__ float4 sh_spheres[BIT_IN_INT*8]; // each thread will load one sphere
    const short thCnt = blockDim.x * blockDim.y;
    const short main_idx = threadIdx.x + threadIdx.y*blockDim.x;
    const int sph_in_last_jump = spheres_cnt % thCnt;
    const int sph_jumps_cnt = spheres_cnt/thCnt + !!sph_in_last_jump;
    for (int curr_rem_idx = main_idx; curr_rem_idx < ints_per_line; curr_rem_idx += thCnt)
    {
        res[curr_rem_idx] = 0;
    }
    __syncthreads();
    for (;z < z_cnt; z++, curr_pnt.z += cell_len)
    {
        for (int curr_jump = 0; curr_jump < jumps; curr_jump++)
        {
            int int_idx = threadIdx.y + blockDim.y * curr_jump;
            curr_pnt.x = (bit_idx + int_idx * 32 + map_shift) * cell_len;
            bool overlapped = (int_idx >= ints_per_line);

            for (int sph_jump = 0; sph_jump < sph_jumps_cnt; sph_jump++)
            {
                int copy_idx = sph_jump*thCnt + main_idx;

                __syncthreads();
                if (copy_idx < spheres_cnt)
                    sh_spheres[main_idx] = spheres[copy_idx];
                __syncthreads();
                int max_idx_in_shared = (sph_jump + 1 == sph_jumps_cnt) ? sph_in_last_jump : thCnt;
                
                int comp_idx = main_idx;
                do
                {
                    if (is_overlapped(curr_pnt, sh_spheres[comp_idx]))
                        overlapped = true;
                    if (++comp_idx >= max_idx_in_shared)
                        comp_idx = 0;
                } while (!overlapped && comp_idx != main_idx);
            }
            if (!overlapped)
            {
#ifdef _SLOW_TEST
				atomicOr(fld + int_idx + blockIdx.x * ints_per_line + z * ints_per_layer, 1 << threadIdx.x);
#else
				atomicOr(&res[int_idx], 1 << threadIdx.x);
#endif
            }
        }
#ifndef _SLOW_TEST
	__syncthreads();
        if (threadIdx.y == 0) {
			for (int curr_int = threadIdx.x; curr_int < ints_per_line; curr_int += blockDim.x)
			{
				atomicOr(fld + curr_int + blockIdx.x * ints_per_line + z * ints_per_layer, res[curr_int]);
				res[curr_int] = 0;
			}
        }
        __syncthreads();
#endif
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
    float shift;
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

    // __shared__ int shifted_line [];
    // shifted_line[x] = templ;
    // if (x == 0)
    // {
    //  shifted_line[blockDim.x] = 0;
    // }
    // __syncthreads();

    if (templ == 0)
    {
        return; // nothing to move
    }

    int bit_shift = 0;
    int first_int = 0;
    int second_int = 0;

    for (int curr_map_idx = 0; curr_map_idx < map.cnt; curr_map_idx++)
    {
        new_y = y + map.y[curr_map_idx];
        new_z = z + map.z[curr_map_idx];
		if (new_y < 0 || gridDim.x <= new_y || new_z < 0 || gridDim.y <= new_z)
			continue;
		// the most tricky part: x axis shift is must be mapped to entire integer
		// shift counted in bits
		int shift = map.x[curr_map_idx];
        if (shift < 0)
        {
            new_x = x + shift / 32 - 1; 
			bit_shift = 32 + shift % 32;
        } else {
			new_x = x + shift / 32;
			bit_shift = shift % 32;
		}
        first_int = templ << bit_shift;
        second_int = templ >> (32 - bit_shift);
        if (0 <= new_x && new_x < ints_per_line)
			atomicOr(result_fld + new_x + new_y * ints_per_line + new_z * ints_per_layer, first_int);
		new_x += 1;
		if (0 <= new_x && new_x < ints_per_line)
			atomicOr(result_fld + new_x + new_y * ints_per_line + new_z * ints_per_layer, second_int);
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

    }  
	if (start_x <= stop_point.x && stop_point.x < stop_x)
    {
        current_val &= 0xFFFFFFFF >> (stop_x - stop_point.x);
    }

	if (current_val == 0) return;
	//atomicAdd(&bits_in_line, NumberOfSetBits(current_val));
 //   __syncthreads();
    atomicAdd(result, NumberOfSetBits(current_val));
}

bool compare_shifts(const iCoord & first, const iCoord & second)
{
    return (first[0] % 32) < (second[0] % 32);
}

CoordVec * get_map(double radius, double sq_len , int divCnt)
{
	CoordVec * result = new CoordVec;
	dCoord centre;
	double centreCoord = sq_len * divCnt / 2.0;
    
    for (int d = 0; d < dCoord::GetDefDims()-1; ++d) {
        centre[d] = centreCoord;
    }
    centre[dCoord::GetDefDims()-1] = radius;
	vector<int> sz(3, divCnt);
    Indexer indx(sz);
    dCoord curr_coord;
    iCoord curr_icoord;
    while (!indx.is_last()) {
        vector<int> curr_vec = indx.curr();
        for (int d = 0; d < curr_vec.size(); ++d) {
            curr_coord[d] = (curr_vec[d] + 0.5) * sq_len;
            curr_icoord[d] = curr_vec[d] - divCnt/2;
        }
        if (is_overlapped(centre, curr_coord)) {
            result->push_back(curr_icoord);
        }
        indx.next();
    }
	return result;
}

void print_map(CoordVec * printed_map)
{
	for (CoordVec::iterator i = printed_map->begin(); i != printed_map->end(); ++i)
	{
		cout << *i << endl;
	}
}

// generates map and stores it in GPU
// for maximum performance map must be sorted (by bit shifts)
Map generate_map(double radius, double cell_len)
// radius – радиус сферы
// a – сторона куба (квадрата)
{
    int divCntSmall = floor(2.0 * radius / cell_len);
	int divCntBig = ceil(2.0 * radius /cell_len);
	int divCnt;
	double cube_vol = pow(cell_len, 3);
	double sph_vol = 4.0/3.0 * PI * pow(radius, 3);
	CoordVec * curr_map;
	if (divCntSmall != divCntBig) 
	{
		CoordVec * big_map = get_map(radius, cell_len, divCntBig);
		double big_size = big_map->size() * cube_vol;
		CoordVec * small_map = get_map(radius, cell_len, divCntSmall);
		double small_size = small_map->size() * cube_vol;

		if (fabs(sph_vol - small_size) < fabs(big_size - sph_vol)) 
		{
			curr_map = small_map;
			delete big_map;
			divCnt = divCntSmall;
		}
		else
		{
			curr_map = big_map;
			delete small_map;
			divCnt = divCntBig;
		}
	} else {
		curr_map = get_map(radius, cell_len, divCntBig);
		divCnt = divCntBig;
	}
    std::sort(curr_map->begin(), curr_map->end(), compare_shifts);
    Map result;
    int gpu_arr_sz = curr_map->size() * sizeof(short);
    cudaSafeCall(cudaMalloc(&(result.x), gpu_arr_sz));
    cudaSafeCall(cudaMalloc(&(result.y), gpu_arr_sz));
    cudaSafeCall(cudaMalloc(&(result.z), gpu_arr_sz));

    short * x = new short[curr_map->size()];
    short * y = new short[curr_map->size()];
    short * z = new short[curr_map->size()];
    for (int idx = 0; idx < curr_map->size(); ++idx)
    {
        x[idx] = curr_map[0][idx][0];
        y[idx] = curr_map[0][idx][1];
        z[idx] = curr_map[0][idx][2];
    }
    cudaSafeCall(cudaMemcpy(result.x, x, gpu_arr_sz, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(result.y, y, gpu_arr_sz, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(result.z, z, gpu_arr_sz, cudaMemcpyHostToDevice));
    result.cnt = curr_map->size();
    result.shift = (divCnt % 2) * 0.5f;

    cout << curr_map->size() << " points in map\n";

    delete [] x;
    delete [] y;
    delete [] z;
    delete curr_map;
    return result;
}

__global__ void init_centers_kernel(int * fld, int ints_per_line, int ignore_bits, int total_ints)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= total_ints) return;
	if (idx % ints_per_line == ints_per_line-1)
		fld[idx] = 0xFFFFFFFF << ignore_bits;
	else
		fld[idx] = 0;
}

void init_centers_fld(int * fld, const dim3 & fld_size)
{
	int ints_per_line = fld_size.x/BIT_IN_INT;
	int ignore_bits = fld_size.x - fld_size.y;
	int total_ints = ints_per_line * fld_size.y * fld_size.z;
	int threads_per_block = 512;
	int num_blocks = total_ints / threads_per_block + 1;
	init_centers_kernel <<< num_blocks, threads_per_block >>>(fld, ints_per_line, ignore_bits, total_ints);
	cudaSafeCall(cudaDeviceSynchronize());
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
    int result = 0xDEADBEEF;


    cudaSafeCall(cudaMalloc(&d_cells_cnt, sizeof(int)));
//    cudaSafeCall(cudaMemset(d_cells_cnt, 0, sizeof(int)));

    cudaSafeCall(cudaMemset(d_fld, 0, total_bytes));
    cout << time_from_start() << ": overlapping... ";

    get_overlapping_field <<<dim_grid_over, dim_block_over>>>
    (d_fld, d_spheres, spheres_cnt, radius, fld_size.z, cell_len, ints_per_line, ints_per_layer,
    iter_map.shift);
    cudaSafeCall(cudaDeviceSynchronize());
    cout << time_from_start() << ": Done\n";
	cout << "In fuction was set: " << fld_bit_cnt(d_fld, total_ints) << endl;
	xor_fields <<<dim_grid_help, dim_block_help>>> (d_fld, d_centers_fld, d_fld, total_ints);
    cudaSafeCall(cudaDeviceSynchronize());
	cout << "After xoring: " << fld_bit_cnt(d_fld, total_ints) << endl;
	/*if (radius*2 > 155)
	{
		cout << "Disable me! Dumping centers\n";
		int * dump = new int[total_ints];
		cudaSafeCall(cudaMemcpy(dump, d_fld, total_bytes, cudaMemcpyDeviceToHost));
		FILE * dump_file = fopen("dump_centers.dmp", "wb");
		fwrite(dump, sizeof(int), total_ints, dump_file);
		fclose(dump_file);
		delete [] dump;
		cout << "Dumped\n";
	}*/
    cout << time_from_start() << ": start apply map... ";
    apply_map <<<dim_grid_map, dim_block_map>>> (d_fld, result_fld, iter_map, ints_per_line, ints_per_layer);
    cudaSafeCall(cudaDeviceSynchronize());
    cout << time_from_start() << ": Done\n";
	result = fld_bit_cnt(result_fld, total_ints);
	cout << "In result fld setted: " << result << endl;
    or_fields <<<dim_grid_help, dim_block_help>>> (d_fld, d_centers_fld, d_centers_fld, total_ints);
    cudaSafeCall(cudaDeviceSynchronize());
    /*cout << time_from_start() << ": counting... " ;
    cudaSafeCall(cudaMemset(d_cells_cnt, 0, sizeof(int)));
	cudaSafeCall(cudaDeviceSynchronize());
    fld_cnt <<<dim_grid_map, dim_block_map>>> (result_fld, start_point, stop_point, d_cells_cnt);
    cudaSafeCall(cudaDeviceSynchronize());
    cout << time_from_start() << ": Done\n";

    cudaSafeCall(cudaMemcpy(&result, d_cells_cnt, sizeof(int), cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaDeviceSynchronize());
*/
	cout << "Result: " << result << endl;
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
    static char * result_filename = new char[256];
    
    double radius = poreSize/2.0;
    printf("Division count = %d\n", divCnt);
	int gpu_points = 27 * h_spheres.size();

    if (result_fld == NULL) {
        // first launch
        size_t total_bits = 1;
        for (int dim = 0; dim < iCoord::GetDefDims(); ++dim) {
            fld_size[dim] = (int)((box.maxCoord[dim] - box.minCoord[dim]) / sq_len + 1);
            zero_pnt[dim] = box.minCoord[dim];
            scale[dim] = sq_len;
        }

        int ignore = (int) poreSize/sq_len/2.0;
        stop_point.x = fld_size[0] - ignore;

        fld_size[0] = iAlignUp(fld_size[0], 32);

        fld_dim.x = fld_size[0];
        fld_dim.y = fld_size[1];
        fld_dim.z = fld_size[2];

		cout << "Fld size: " << fld_size << endl;

        for (int dim = 0; dim < iCoord::GetDefDims(); ++dim) {
            total_bits *= fld_size[dim];
        }

        start_point.x = ignore;
        start_point.y = ignore;
        start_point.z = ignore;

        stop_point.y = fld_dim.y - ignore;
        stop_point.z = fld_dim.z - ignore;

        scale[iCoord::GetDefDims()] = 1; // scale of radius
        size_t total_bytes = (size_t)total_bits/8;
        cout << "Need memory: " << total_bytes * 3 << endl;
        cudaSafeCall(cudaMalloc(&result_fld, total_bytes));
        cudaSafeCall(cudaMemset(result_fld, 0, total_bytes));
		//init_centers_fld(result_fld, fld_dim);
		//prev_cells_cnt = fld_bit_cnt(result_fld, total_bytes/4);
        cudaSafeCall(cudaMalloc(&centers_fld, total_bytes));
		cudaSafeCall(cudaMemset(centers_fld, 0, total_bytes));
		//init_centers_fld(centers_fld, fld_dim);
        cudaSafeCall(cudaMalloc(&curr_centers_fld, total_bytes));
        cudaSafeCall(cudaMemcpy(curr_centers_fld, centers_fld, total_bytes, cudaMemcpyDeviceToDevice));

        // send to GPU
		// make periodic conditions 
		// in that case we have 27 times more points
		// it's bad, but let's try
		// moreover we can optimize it later
		// TODO: make personal array of points for each block
		
        float4 * h_spheres_floats = new float4[gpu_points];
		Indexer shifts(vector<int>(3, 3));
		int floats_idx = 0;

        for (int i = 0; i < h_spheres.size(); ++i)
        {
			while (!shifts.is_last())
			{
				vector<int> curr_shifts = shifts.curr();
				for (vector<int>::iterator sh = curr_shifts.begin(); sh != curr_shifts.end(); ++sh)
				{
					*sh -= 1;
				}
				h_spheres_floats[floats_idx++] = make_float4(curr_shifts[0] * box.maxCoord[0] + h_spheres[i][0],
					curr_shifts[1] * box.maxCoord[1] + h_spheres[i][1], 
					curr_shifts[2] * box.maxCoord[2] + h_spheres[i][2], h_spheres[i][3]);
				shifts.next();
			}
			shifts.to_begin();
        }
		if (floats_idx != gpu_points)
		{
			std::cerr << "Strange floats cnt: " << floats_idx << " ins " << gpu_points << endl;
			exit(40);
		}
        cudaSafeCall(cudaMalloc(&d_spheres, gpu_points * sizeof(float4)));
        cudaSafeCall(cudaMemcpy(d_spheres, h_spheres_floats, gpu_points * sizeof(float4), 
                    cudaMemcpyHostToDevice));
        delete [] h_spheres_floats;
        sprintf(result_filename, "psd_%d.txt", rand());
    }
    
    int cells_cnt = iteration(d_spheres, gpu_points, curr_centers_fld, centers_fld, 
                                fld_dim, radius, sq_len, result_fld, 
                                start_point, stop_point);

	ofstream fout(result_filename, ios_base::app);
	fout << radius*2 << '\t' << cells_cnt << endl;
	fout.close();
    cout << "Result written in " << result_filename << endl;
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
    if( access( fname, 0 ) != -1 ) {
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
    cout << "Profiled version\n";
    srand(time(NULL));
	/*int my_argc = 5;
	char ** my_argv = new char *[my_argc];
	for (int i = 0; i < my_argc; ++i)
	{
		my_argv[i] = new char[256];
	}
	strcpy(my_argv[0], argv[0]);
	strcpy(my_argv[1], "testing.dat");
	strcpy(my_argv[2], "4");
	strcpy(my_argv[3], "5.4641016");
	strcpy(my_argv[4], "-float");*/
    TCalcPlan plan;
	int plan_error = plan.Init(argc, argv);
	if (plan_error)
	{
		int i;
		scanf("%d", &i);
		return plan_error;
	}
	int divisions = plan.divisions;
    cudaSafeCall(cudaSetDevice(plan.gpu));
    time_from_start();
    
    if (plan.is_float) {
        load_coords<float>(plan.filename, &v);
    } else {
        load_coords<double>(plan.filename, &v);
    }
    
	vector<double> distr = getDistribution(v, plan.min_r, plan.max_r, plan.step, divisions, plan.field_size);
    for (vector<double>::reverse_iterator curr_d = distr.rbegin(); curr_d != distr.rend(); ++curr_d) {
        cout << *curr_d << " ";
    }
    cout << endl;
//	int prevent_exit = 0;
//    std::cin >> prevent_exit;
	return 0;
}

