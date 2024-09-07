# define BOOST_PYTHON_STATIC_LIB
# define BOOST_NUMPY_STATIC_LIB
# include<boost/python/numpy.hpp>
# include<boost/python.hpp>
# include<boost/python/suite/indexing/vector_indexing_suite.hpp>
# include<iostream>
# include<vector>
# include<thread>
# include<mutex>


namespace p =  boost::python;
namespace np = boost::python::numpy;


float* Pixel_Wise_Sampler(int row, int column, int radius, int* HWC, int res_num, int* resolutions, float* map)
{
	int height = HWC[0];
	int width = HWC[1];
	int channel = HWC[2];

	// 根据r和c确定采样点中心
	int ws = 2 * radius + 1;

	float* sample_patch = new float[ws * ws * channel * res_num];

	// 初始化指针数组，理论上此处并不需要该步骤，因为后面会重新赋值
	for (int m = 0; m < ws * ws * channel * res_num; m++)
	{
		sample_patch[m] = 0;
	}

	// 首先是计算分辨率，用于确定裁剪范围
	for (int res_idx = 0; res_idx < res_num; res_idx++) {
		// 基础图的分辨率为10米，因此除以10转为像素数量
		int res = resolutions[res_idx] / 10;

		// 依次计算各个块的范围
		for (int r = -radius; r < radius + 1; r++) {
			for (int c = -radius; c < radius + 1; c++) {
				// 确定当前范围，即左-上点的横纵坐标，则整个范围采样范围实际上就是 -> [sub_ldcr: sub_ldcr + res, sub_ldcc: sub_ldcc + res]
				int sub_ldcr = row + r * res - (res - 1) / 2;
				int sub_ldcc = column + c * res - (res - 1) / 2;

				for (int cd = 0; cd < channel; cd++) {
					// 通道数量，逐个通道计算，此处计划的patch应当是[res_num, channel, ws, ws]
					// 当前点在patch中的一维坐标
					int patch_idx = ws * ws * (channel * res_idx + cd) + (r + radius) * ws + (c + radius);
					double count = 0;

					// 在当前这个[res, res]块内计算均值，注意这次用的double型，是为了防止一些浮点计算的错误
					for (int sr = 0; sr < res; sr++) {
						for (int sc = 0; sc < res; sc++) {
							int index = height * width * cd + (sub_ldcr + sr) * width + sub_ldcc + sc;
							double v = (double)map[index];
							count += v;
						}
					}
					// 赋值
					sample_patch[patch_idx] = (float)(count / (res * res));
				}
			}
		}

	}

	return sample_patch;
}


float* convert_float_3d_to_1d(np::ndarray& arr) {
	int height = arr.shape(0);
	int width = arr.shape(1);
	int channel = arr.shape(2);
	printf("TAG1\n");
	float* data = reinterpret_cast<float*>(arr.get_data());
	float* sorted_data = new float[height * width * channel];
	printf("TAG2\n");

	int count = 0;

	for (int c = 0; c < channel; c++)
	{
		for (int i = 0; i < height; ++i)
		{
			for (int j = 0; j < width; ++j)
			{
				int idx = i * width * channel + j * channel + c;
				sorted_data[count] = data[idx];
				count++;
			}
		}
	}

	//delete[] data;
	//data = NULL;
	printf("TAG3\n");
	return sorted_data;
}


int* convert_int_2d_to_1d(np::ndarray& arr) {
	int height = arr.shape(0);
	int width = arr.shape(1);

	int* data = reinterpret_cast<int*>(arr.get_data());

	int* sorted_data = new int[height * width];

	int count = 0;

	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			int idx = i * width + j;
			sorted_data[count] = data[idx];
			count++;
		}
	}

	//delete[] data;
	//data = NULL;

	return sorted_data;
}


int* convert_int_1d_to_1d(np::ndarray& arr) {
	int length = arr.shape(0);

	int* data = reinterpret_cast<int*>(arr.get_data());

	int* sorted_data = new int[length];

	for (int i = 0; i < length; ++i)
	{
		sorted_data[i] = data[i];
	}

	//delete[] data;
	//data = NULL;

	return sorted_data;
}


int* GetShape3d(np::ndarray& arr) {
	int* shape = new int[3];
	shape[0] = arr.shape(0);
	shape[1] = arr.shape(1);
	shape[2] = arr.shape(2);

	return shape;
}

int* GetShape2d(np::ndarray& arr) {
	int* shape = new int[2];
	shape[0] = arr.shape(0);
	shape[1] = arr.shape(1);

	return shape;
}

int* GetShape1d(np::ndarray& arr) {
	int* shape = new int[1];
	shape[0] = arr.shape(0);
	return shape;
}

np::ndarray float_ptr_to_3d_ndarray(float* data, int dim_x, int dim_y, int dim_z)
{
	p::tuple shape = p::make_tuple(dim_x, dim_y, dim_z);

	const int stride_y = sizeof(float);
	const int stride_x = dim_y * stride_y;
	const int stride_z = dim_x * stride_x;

	p::tuple strides = p::make_tuple(stride_x, stride_y, stride_z);

	np::dtype dt = np::dtype::get_builtin<float>();
	np::ndarray array = np::from_data(data, dt, shape, strides, p::object());
	// 使用 PyCapsule 创建一个 owner 对象
	PyObject* capsule = PyCapsule_New(data, NULL, [](PyObject* capsule) {
		float* data = static_cast<float*>(PyCapsule_GetPointer(capsule, NULL));
		delete[] data;
		});
	p::handle<> handle(capsule);
	p::object owner(handle);
	array.set_base(owner);
	return array;
}


np::ndarray RC_convert(int* rc_pair)
{
	p::tuple shape = p::make_tuple(2);
	const int stride = sizeof(int);

	p::tuple strides = p::make_tuple(stride);

	np::dtype dt = np::dtype::get_builtin<int>();
	np::ndarray array = np::from_data(rc_pair, dt, shape, strides, p::object());
	// 使用 PyCapsule 创建一个 owner 对象
	PyObject* capsule = PyCapsule_New(rc_pair, NULL, [](PyObject* capsule) {
		int* data = static_cast<int*>(PyCapsule_GetPointer(capsule, NULL));
		delete[] data;
		});
	p::handle<> handle(capsule);
	p::object owner(handle);
	array.set_base(owner);
	return array;
}


void each_thread(int start_idx, int end_idx, int radius, float* map, int* HWC, int* resolutions, int* RN, int* valid_coords,
	             std::vector<float*>& results, std::mutex& result_mut, std::vector<int*>& RowCol) 
{
	for (int i = start_idx; i < end_idx; i++)
	{
		int row = valid_coords[i * 2];
		int col = valid_coords[i * 2 + 1];
		float* patch = Pixel_Wise_Sampler(row, col, radius, HWC, RN[0], resolutions, map);

		std::lock_guard<std::mutex> guard(result_mut);
		results.push_back(patch);

		int* rc_pair = new int[2];
		rc_pair[0] = row;
		rc_pair[1] = col;
		RowCol.push_back(rc_pair);
	}
}


p::tuple S2R10_MCHR(np::ndarray& map, np::ndarray& coords, np::ndarray& resolutions, int Radius, int ms_ratio) {
	p::list PList;
	p::list RC_List;

	float* m = convert_float_3d_to_1d(map);
	printf("TAG1");
	int* c = convert_int_2d_to_1d(coords);
	printf("TAG2");
	int* rs = convert_int_1d_to_1d(resolutions);
	printf("TAG3");
	
	int* HWC = GetShape3d(map);
	int* C2 = GetShape2d(coords);
	int* RN = GetShape1d(resolutions);

	printf("The shape of map is: height: %d, width: %d, channel: %d. \n", HWC[0], HWC[1], HWC[2]);
	
	int nums = C2[0];
	
	const int num_threads = std::thread::hardware_concurrency();
	//const int activate_threads = (const int)num_threads * ms_ratio;
	const int iters = nums;
	int work_per_thread = iters / num_threads;

	std::mutex patches_mutex;
	std::vector<float*> patches;
	std::vector<std::thread> threads;
	std::vector<int*> ROWCOL;

	printf("The total number of patches is: %d. \n", nums);
	
	for (int thread_idx = 0; thread_idx < num_threads; thread_idx++)
	{
		int start = thread_idx * work_per_thread;
		int end = (thread_idx == num_threads - 1) ? iters : (thread_idx + 1) * work_per_thread;
		// parameter-> int start_idx, int end_idx, int radius, float* map, int* HWC, int* resolutions, int* RN, int* valid_coords,
		//             std::vector<float*>& results, std::mutex& result_mut
		threads.emplace_back(each_thread, start, end, Radius, m, HWC, rs, RN, c, std::ref(patches), std::ref(patches_mutex), std::ref(ROWCOL));
	}

	for (auto& thread : threads)
	{
		thread.join();
	}

	threads.clear();
	threads.shrink_to_fit();
	
	printf("The Sampling is finished. \n");
	
	for(int idx = 0; idx < nums; idx++)
	{
		np::ndarray element = float_ptr_to_3d_ndarray(patches[idx], Radius * 2 + 1, Radius * 2 + 1, HWC[2] * RN[0]);
		np::ndarray row_col = RC_convert(ROWCOL[idx]);
		RC_List.append(row_col);
		PList.append(element);
	}
	
	// 清空内存
	delete[] m;
	m = NULL;
	delete[] c;
	c = NULL;
	delete[] rs;
	rs = NULL;
	delete[] HWC;
	HWC = NULL;
	delete[] C2;
	C2 = NULL;
	delete[] RN;
	RN = NULL;

	

	return p::make_tuple(PList, RC_List);
}


BOOST_PYTHON_MODULE(S2R10_MCHR)
{
	Py_Initialize();
	np::initialize();

	def("S2R10_MCHR", S2R10_MCHR);
}

//     g++ -shared -fPIC -O2 -I/usr/include/python3.8 -I/usr/include/boost S2R10_MCHR.cpp -o S2R10_MCHR.so -lboost_python38 -lboost_numpy38
