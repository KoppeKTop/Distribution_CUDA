
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Новая версия
// Что будет делать?
// В общем массиве на ГПУ будут храниться координаты сфер структуры
// В едином битовом массиве будут храниться уже отмеченные точки
#include <stdio.h>

#ifdef __linux__
#include <unistd.h>
#else
#include <io.h>
#endif
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <map>
#include <utility>
#include <algorithm>
#include "Indexer.h"
#include "Coord.h"
#include "cuda_helper.h"
#include "CalcPlan.h"
#include <cuda_runtime.h>
#include "Rect.h"
#include <fstream>
#include <string>

#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/logical.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

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


struct min_dist_point_to_sph
{
	const float4 sph;
	const dim3 fld_sz;
	const float cell_len;
	const float map_shift;

	min_dist_point_to_sph(const float4 c_sph, const dim3 c_fld_sz,
			const float c_cell_len, const float c_map_shift):
		sph(c_sph),
		fld_sz(c_fld_sz),
		cell_len(c_cell_len),
		map_shift(c_map_shift)
	{}

	__host__ __device__ float sqr(const float & x) const {
		return x*x;
	}

	__host__ __device__ float operator()(const float & prev_dist, const int & coord_index) const
	{
		int x, y, z;
		float curr_dist = 0;
		z = coord_index / (fld_sz.x * fld_sz.y);
		y = (coord_index - z * fld_sz.x * fld_sz.y) / fld_sz.x;
		x = coord_index - z * fld_sz.x * fld_sz.y - y * fld_sz.x;

		curr_dist += sqr((x + map_shift) * cell_len - sph.x);
		curr_dist += sqr((y + map_shift) * cell_len - sph.y);
		curr_dist += sqr((z + map_shift) * cell_len - sph.z);
		curr_dist = sqrtf(curr_dist);
		curr_dist -= sph.w;
		if (curr_dist < prev_dist) {
			return curr_dist;
		}
		return prev_dist;
	}
};


void expand_spheres(const SphereVec & h_spheres, const Rect & box, vector<float4> & res) {
	res.clear();
	Indexer shifts(vector<int>(3, 3));

	for (int i = 0; i < h_spheres.size(); ++i)
	{
		while (!shifts.is_last())
		{
			vector<int> curr_shifts = shifts.curr();
			for (vector<int>::iterator sh = curr_shifts.begin(); sh != curr_shifts.end(); ++sh)
			{
				*sh -= 1;
			}
			shifts.next();
			if (curr_shifts[0] == 0 && curr_shifts[1] == 0 &&
					curr_shifts[2] == 0)
			{
				continue;
			}
			res.push_back(make_float4(curr_shifts[0] * box.maxCoord[0] + h_spheres[i][0],
					curr_shifts[1] * box.maxCoord[1] + h_spheres[i][1],
					curr_shifts[2] * box.maxCoord[2] + h_spheres[i][2], h_spheres[i][3]));
		}
		shifts.to_begin();
	}
}

void shrink_spheres(vector<float4>::const_iterator start, vector<float4>::const_iterator end, const float radius, const Rect & box, vector<float4> & res)
{
	res.clear();
	for (auto sph = start; sph != end; ++sph) {
		if (sph->x + sph->w + radius < box.minCoord[0]) continue;
		if (sph->y + sph->w + radius < box.minCoord[1]) continue;
		if (sph->z + sph->w + radius < box.minCoord[2]) continue;
		if (sph->x - sph->w - radius > box.maxCoord[0]) continue;
		if (sph->y - sph->w - radius > box.maxCoord[1]) continue;
		if (sph->z - sph->w - radius > box.maxCoord[2]) continue;
		res.push_back(*sph);
	}
}

float get_dist_field(const SphereVec & h_spheres, const dim3 fld_sz, const float cell_len,
		const float map_shift, thrust::device_vector<float> & dist_fld)
{
	float max_dist = sqrtf(SQR(fld_sz.x) + SQR(fld_sz.y) + SQR(fld_sz.z)); // diagonal is a max distance
	vector<float4> orig_sph, expanded_sph, shrinked_sph;
	// first of all original spheres will be in the list
	for (auto sph : h_spheres) {
		orig_sph.push_back(make_float4(sph[0], sph[1], sph[2], sph[3]));
	}

	Rect box;
	for (int i = 0; i < 3; i++)
		box.minCoord[i] = 0;
	box.maxCoord[0] = fld_sz.x * cell_len;
	box.maxCoord[1] = fld_sz.y * cell_len;
	box.maxCoord[2] = fld_sz.z * cell_len;

	thrust::fill(dist_fld.begin(), dist_fld.end(), max_dist);
	thrust::counting_iterator<int> cnt(0);
	for (auto curr_sph : orig_sph) {
		min_dist_point_to_sph dist_calc(curr_sph, fld_sz, cell_len, map_shift);
		thrust::transform(dist_fld.begin(), dist_fld.end(), cnt, dist_fld.begin(), dist_calc);
	}
	cout << "Finished internal spheres" << endl;
	float max_radius = *thrust::max_element(dist_fld.begin(), dist_fld.end());
	cout << "Maximum radius: " <<  max_radius << endl;
	expand_spheres(h_spheres, box, expanded_sph);
	shrink_spheres(expanded_sph.begin(), expanded_sph.end(), max_radius, box, shrinked_sph);
	cout << shrinked_sph.size() << " external spheres from " << expanded_sph.size() << endl;
	for (auto curr_sph : shrinked_sph) {
		min_dist_point_to_sph dist_calc(curr_sph, fld_sz, cell_len, map_shift);
		thrust::transform(dist_fld.begin(), dist_fld.end(), cnt, dist_fld.begin(), dist_calc);
	}
	max_radius = *thrust::max_element(dist_fld.begin(), dist_fld.end());
	cout << "Dist map done. Final maximum radius: " << max_radius << endl;
    return max_radius;
}


struct mark_point
{
	const float radius;
	const dim3 fld_sz;
	char * res_fld;
	const int4 delta;

	mark_point(const float c_radius, const dim3 c_fld_sz,
			char * c_res_fld, const int4 c_delta):
		radius(c_radius),
		fld_sz(c_fld_sz),
		res_fld(c_res_fld),
		delta(c_delta)
	{}

	__host__ __device__ int toroise_coord(int coord, int max_coord) const
	{
		if (coord < 0) {
			return coord + max_coord;
		} else if (coord >= max_coord) {
			return coord - max_coord;
		}
		return coord;
	}

	__host__ __device__ char operator()(const float & distance, const int & coord_index) const
	{
		if (distance <= radius) {
			return 0;
		}
		int x, y, z, new_idx;

		z = coord_index / (fld_sz.x * fld_sz.y);
		y = (coord_index - z * fld_sz.x * fld_sz.y) / fld_sz.x;
		x = coord_index - z * fld_sz.x * fld_sz.y - y * fld_sz.x;

		x = toroise_coord(x + delta.x, fld_sz.x);
		y = toroise_coord(y + delta.y, fld_sz.y);
		z = toroise_coord(z + delta.z, fld_sz.z);

		for (int d = 0; d < delta.w; ++d) {
			new_idx = x + fld_sz.x * y + fld_sz.x * fld_sz.y * z;
			res_fld[new_idx] = 1;
			x = toroise_coord(x + 1, fld_sz.x);
		}

		return 1;
	}
};

void coalesce_map(const CoordVec * curr_map, vector<int4> & coalesced)
{
	typedef std::map<std::pair<int, int>, int4> res_map;
	res_map result;
	for (size_t map_idx = curr_map->size(); map_idx != 0; map_idx -- )
	{
		const iCoord curr_coord = curr_map->at(map_idx-1);
		const std::pair<int, int> curr_id(curr_coord[1], curr_coord[2]);
		if (result.find(curr_id) != result.end()) {
			int4 curr_descr = result[curr_id];
			if (curr_descr.x > curr_coord[0]) {
				curr_descr.x = curr_coord[0];
			}
			curr_descr.w += 1;
			result[curr_id] = curr_descr;
		} else {
			int4 curr_descr = make_int4(curr_coord[0], curr_coord[1], curr_coord[2], 1);
			result[curr_id] = curr_descr;
		}
	}
	coalesced.clear();
	for (res_map::iterator map_it = result.begin(); map_it != result.end(); ++map_it) {
		coalesced.push_back(map_it->second);
	}
}

void mark_map(const thrust::device_vector<float> & distances, const CoordVec * curr_map, const float radius,
		const dim3 fld_sz, char * d_result)
{
	thrust::device_vector<char> centers_fld(distances.size());
	vector<int4> coalesced_map;
	coalesce_map(curr_map, coalesced_map);

	cout << "Apply map for radius " << radius << ", " << coalesced_map.size() << " Points in map" << endl;

	int status_print = coalesced_map.size() / 10;
	if (status_print < 10) {
		status_print = 100;
	}

	for (size_t map_idx = 0; map_idx < coalesced_map.size(); ++map_idx)
	{
		int4 delta = coalesced_map[map_idx];
		mark_point op(radius, fld_sz, d_result, delta);
		thrust::counting_iterator<int> cnt(0);
		thrust::transform(distances.begin(), distances.end(), cnt, centers_fld.begin(), op);

		if ((map_idx + 1) % status_print == 0) {
			cout << map_idx + 1 << " points done" << endl;
		}
	}

	cout << "Finish apply map" << endl;
}

void mark_map(const thrust::host_vector<float> & distances, const CoordVec * curr_map, const float radius,
		const dim3 fld_sz, char * d_result)
{
	thrust::host_vector<char> centers_fld(distances.size());
	vector<int4> coalesced_map;
	coalesce_map(curr_map, coalesced_map);

	cout << "Apply map for radius " << radius << ", " << coalesced_map.size() << " Points in map\n";

	int status_print = coalesced_map.size() / 100;

	for (size_t map_idx = 0; map_idx < coalesced_map.size(); ++map_idx)
	{
		int4 delta = coalesced_map[map_idx];
		mark_point op(radius, fld_sz, d_result, delta);
		thrust::counting_iterator<int> cnt(0);
		thrust::transform(thrust::host, distances.begin(), distances.end(), cnt, centers_fld.begin(), op);

		if ((map_idx + 1) % status_print == 0) {
			cout << map_idx + 1 << " points done" << endl;
		}
	}

	cout << "Finish apply map" << endl;
}


bool is_overlapped(const dCoord & sph1, const dCoord & sph2)
{
    float r_sum = SQR(float(sph1[3] + sph2[3]));
    float r = SQR((float)(sph1[0] - sph2[0])) + 
    SQR(float(sph1[1] - sph2[1])) + SQR(float(sph1[2] - sph2[2]));
    return ((r - r_sum) < float(1e-4));
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

CoordVec * generate_host_map(const double radius, const double cell_len, int & divCnt)
{
	int divCntSmall = floor(2.0 * radius / cell_len);
	int divCntBig = ceil(2.0 * radius /cell_len);
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

	return curr_map;
}

template <typename T>
struct gt
{
	T rhs;
	gt(const T c_rhs):
		rhs(c_rhs)
	{}

	__host__ __device__ bool operator()(const T& lhs) { return lhs > rhs; }
};


void dump_maps(const thrust::host_vector<float> & dist_fld , const thrust::host_vector<float> & dist_fld_shift)
{
	char buf[256];
	sprintf(buf, "%d.maps", rand());

	const float * fld_1 = thrust::raw_pointer_cast(&dist_fld[0]);
	const float * fld_2 = thrust::raw_pointer_cast(&dist_fld_shift[0]);

	FILE * f = fopen(buf, "wb");
	if (f == NULL) {
		cout << "Can not open file " << buf << endl;
		return;
	}
	fwrite(fld_1, sizeof(float), dist_fld.size(), f);
	fwrite(fld_2, sizeof(float), dist_fld_shift.size(), f);
	fclose(f);
	cout << "Maps saved in " << buf << endl;
}

float load_maps(const char * fn, thrust::host_vector<float> & dist_fld , thrust::host_vector<float> & dist_fld_shift)
{
	FILE * f = fopen(fn, "rb");
	if (f == NULL) {
		cout << "Cannot open map file " << fn  << endl;
		return 0;
	}
	fseek(f, 0L, SEEK_END);
	size_t sz = ftell(f);
	rewind(f);
	size_t cnt = sz / sizeof(float) / 2;
	size_t ret;
	float * buffer = new float[cnt];
	ret = fread(buffer, sizeof(float), cnt, f);
	if (ret != cnt) {
		cout << "bad maps file\n";
		return 0;
	}
	dist_fld.assign(buffer, buffer+cnt);
	ret = fread(buffer, sizeof(float), cnt, f);
	if (ret != cnt) {
		cout << "bad maps file\n";
		return 0;
	}
	dist_fld_shift.assign(buffer, buffer+cnt);

	fclose(f);
	cout << "Loaded maps. Point count: " << cnt << endl;
    float max_radius = *thrust::max_element(dist_fld.begin(), dist_fld.end());
	cout << "No shift. Max: " << max_radius << ", min: "
			<< *thrust::min_element(dist_fld.begin(), dist_fld.end()) << endl;
    float curr_r = *thrust::max_element(dist_fld_shift.begin(), dist_fld_shift.end());
	cout << "Shift. Max: " << curr_r << ", min: "
				<< *thrust::min_element(dist_fld_shift.begin(), dist_fld_shift.end()) << endl;
    if (curr_r > max_radius) max_radius = curr_r;
	return max_radius;
}

#define RUN_DEVICE

void getDistribution(const SphereVec & spheres, double minPores, double maxPores, double h, int divisions,
        double field_size, const std::string & maps_fn = "")
{
	double sq_len = min(h, minPores/divisions);
	int fld_dim = ceil(field_size/ sq_len);
	dim3 fld_sz(fld_dim, fld_dim, fld_dim);
	size_t fld_elements = fld_sz.x*fld_sz.y*fld_sz.z;

	char * result = NULL;
#ifdef RUN_DEVICE
	cudaSafeCall(cudaMalloc(&(result), fld_elements));
	cudaSafeCall(cudaMemset(result, 0, fld_elements));
#else
	result = new char[fld_elements];
	memset(result, 0, fld_elements);
#endif

	vector<int> res_psd;
	thrust::device_vector<float> d_dist_fld(fld_elements);
	thrust::device_vector<float> d_dist_fld_shift(fld_elements);
	thrust::host_vector<float>  h_dist, h_dist_shift;
    float max_radius = 0;
	if (maps_fn.length() > 1) {
		max_radius = load_maps(maps_fn.c_str(), h_dist, h_dist_shift);
		d_dist_fld = h_dist;
		d_dist_fld_shift = h_dist_shift;
	} else {
		max_radius = get_dist_field(spheres, fld_sz, sq_len, 0, d_dist_fld);
		float curr_r = get_dist_field(spheres, fld_sz, sq_len, 0.5, d_dist_fld_shift);
        if (curr_r > max_radius) max_radius = curr_r;
		h_dist = d_dist_fld;
		h_dist_shift = d_dist_fld_shift;
#ifdef DEBUG
		dump_maps(h_dist, h_dist_shift);
#endif
	}

	for (double curr_d = maxPores; curr_d >= minPores; curr_d -= h)
	{
		double radius = curr_d/2;
        if (radius > max_radius) {
            res_psd.push_back(0);
            continue;
        }
		int divCnt;
		CoordVec * curr_map = generate_host_map(radius, sq_len, divCnt);
#ifdef RUN_DEVICE
		if (divCnt % 2 == 0) {
			mark_map(d_dist_fld, curr_map, radius, fld_sz, result);
		}
		else {
			mark_map(d_dist_fld_shift, curr_map, radius, fld_sz, result);
		}
#else
		if (divCnt % 2 == 0) {
			mark_map(h_dist_fld, curr_map, radius, fld_sz, result);
		} else {
			mark_map(h_dist_fld_shift, curr_map, radius, fld_sz, result);
		}
#endif

		int curr_cnt = 0;
#ifdef RUN_DEVICE
		thrust::device_ptr<char> dev_ptr(result);
		curr_cnt = thrust::count(dev_ptr, dev_ptr + fld_elements, (char)1);
#else
		vector<char> host_vec(result, result + fld_elements);
		curr_cnt = thrust::count(host_vec.begin(), host_vec.end(), (char)1);
#endif
		res_psd.push_back(curr_cnt);
		cout << "For radius " << radius << " occupied " << curr_cnt << " cells\n";
		delete curr_map;
	}

#ifdef RUN_DEVICE
	cudaSafeCall(cudaFree(result));
#else
	delete [] result;
#endif
	result = NULL;

	int max_fill = thrust::count_if(d_dist_fld_shift.begin(), d_dist_fld_shift.end(), gt<float>(sq_len / 2));
	res_psd.push_back(max_fill);
	cout << "Max filling: " << max_fill << endl;

	cout << "Final results:\n";
	for (size_t pore_idx = res_psd.size(); pore_idx != 0; pore_idx--)
	{
		cout << res_psd[pore_idx-1]/(double)fld_elements << endl;
	}

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

    getDistribution(v, plan.min_r, plan.max_r, plan.step, divisions, plan.field_size, plan.maps_fn);

	return 0;
}

