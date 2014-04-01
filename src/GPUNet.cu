/*
 * GPUNet.cpp
 *
 *  Created on: Jan 5, 2014
 *      Author: trevor
 *
 *  GPU Neural Network
 *  Maintains network state and invokes functions on the GPU
 *
 */

#include "GPUNet.h"
#include "NetTrainer.h"
#include "NetIO.h"
#include <boost/lexical_cast.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <thrust/device_vector.h>
#include "curand_kernel.h"

/*
 * ------------ CUDA ------------
 */


/**
 * Get a random number within a given float range
 * \param min float
 * \param max float
 * \param i int
 * \param *global curandState
 */
__device__ float get_random_range(float min, float max, int i, curandState *global) {
	curandState local = global[i];
	float r = curand_uniform(&local);
	global[i] = local;
	return min + r * (max - min);
}

__device__ float get_ih_weight(float* ih_weights, int n_hidden, int i, int h) {
	return ih_weights[n_hidden*i + h];
}

__device__ float get_ho_weight(float* ho_weights, int n_output, int h, int o) {
	return ho_weights[n_output*h + o];
}

/**
 * Compute the sigmoid value of a given float
 * \param x the value to compute the sigmoid of
 */
__device__ inline float sigmoid(float x) {
	return 1.0 / (1.0 + exp(-x));
}


/**
 * Compute the output gradient given specific output and target values
 * \param output float
 * \param target float
 */
__device__ float calc_output_gradient(float output, float target) {
	return output * (1 - output) * (target - output);
}


/**
 * Clamp the output to 0 or 1 if within .1
 *\param f the value to clamp
 */
__device__ int clamp(float f) {
	if (f < .1) {
		return 0;
	} else if (f > .9) {
		return 1;
	} else {
		return -1;
	}
}

/*
 *
 * ------------- Initialization kernels ---------------
 *
 */


/**
 * Initialize random seeds in CUDA, will initalize blocksize seeds
 */
__global__ void curand_setup(curandState *state) {
	unsigned int seed = (unsigned int)clock64();
	int id = threadIdx.x;
	curand_init(seed, id, 0, &state[id]);
}

/**
 * initialize nodes to 0 or 1 if bias
 * generic
 */
__global__ void init_nodes_layer_v2(int n, float *nodes) {
	unsigned int i = blockIdx.x * blockDim.x+threadIdx.x;
	if (i < n) {
		if (i == n-1)
			nodes[i] = 1;
		else
			nodes[i] = 0;
	}
}

/**
 * set all output nodes to 0
 */

__global__ void init_nodes_output_v2(int n, float *output) {
	unsigned int i = blockIdx.x * blockDim.x+threadIdx.x;
	if (i < n) {
		output[i] = 0;
	}
}


__global__ void init_weights_v2(int n1, int n2, float *weights, curandState *state) {
	unsigned int i = blockIdx.x * blockDim.x+threadIdx.x;
	// r is the range for random values
	if (i < (n1+1)*n2) {
		float r = 1.0 / sqrt((float)blockDim.x-1);
		int node_l1 = i % (n1+1);
		int node_l2 = i % n2;
		weights[n2*node_l1 + node_l2] = get_random_range(-r, r, threadIdx.x, state);
	}
}


__global__ void init_deltas_v2(unsigned int n1, unsigned int n2, float *deltas) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < (n1+1)*n2) {
		deltas[i] = 0;
	}
}



/* --------------- Referencing and simple set function ---------------
 * set bias
 *
 */

//used when copying patterns to device
__global__ void set_bias(int n_input, float *d_inp) {
	d_inp[n_input] = 1;
}


/*
 * -------------- Error calculation ---------------
 * output_correct
 * mse_sum
 *
 */

__device__ int d_num_correct = 0;
__device__ float d_acc = 0;
__device__ float d_mse_sum = 0;
__device__ float d_mse = 0; //current mse

__global__ void output_correct_v2(float *output, float *target, int n_output) {
	for (int i = 0; i < n_output; ++i) {
		if (clamp(output[i]) != clamp(target[i])) {
			return;
		}
	}
	++d_num_correct;
}

__global__ void calc_acc(int n_patterns) {
	d_acc = ((float)d_num_correct/n_patterns * 100);
	d_num_correct = 0;
}

/**
 * single threaded
 */
__global__ void mse_sum_v2(float *output, float *target, int n_output) {
	float sum = 0;
	for (int i = 0; i < n_output; ++i) {
		sum += pow(output[i] - target[i], 2);
	}
	d_mse_sum += sum;
}

/**
 * single threaded
 */
__global__ void calc_mse(int n_output, int n_patterns) {
	d_mse = d_mse_sum / (n_output * n_patterns);
	d_mse_sum = 0;
}



/*
 * ---- feed forward kernels -----------
 *
 * method 1 calculates each node in the next layer with a single thread computing for each output node
 * method 2 has a thread for each term in the linear combination to compute the output
 *     then the activation is computed after syncing threads.
 */

/*
 * to measure bandwidth:
 * (bytes read + bytes writen) / (time secs * 10^9) = gb
 *
 * bytes read = 4* ((n_layer1+1)*2),
 * bytes written = 4* (n_layer2)
 * total/thread = 4*((n_layer1+1)*2 + n_layer2)
 * threads l1 -> l2 = n_hidden
 * threads l2 -> l3 = n_output
 *
 * total_l1->l2 = n_hidden*4*((n_layer1+1)*2 + n_layer2)
 * total_l2->l3 = n_output*4*((n_layer2+1)*2 + n_layer3)
 *
 * total = total_l1->l2 + total_l2->l3;
 */
__global__ void feed_forward_layer_v1(int n_layer1, int n_layer2, float* layer1, float* layer2, float* weights) {
	int n = threadIdx.x; // node to compute;

	float r = 0;
	for (int i = 0; i <= n_layer1; ++i) { //include bias
		r += layer1[i] * weights[n_layer2*i + n];
	}
	layer2[n] = sigmoid(r);
}

/*
 * Generic version, called with pow of 2 threads
 */
__global__ void feed_forward_layer_v1_2(int n_layer1, int n_layer2, float* layer1, float* layer2, float* weights) {
	unsigned int n = blockIdx.x * blockDim.x+threadIdx.x; // node to compute;
	if (n < n_layer2) {
		float r = 0;
		for (int i = 0; i <= n_layer1; ++i) { //include bias
			r += layer1[i] * weights[n_layer2*i + n];
		}
		layer2[n] = sigmoid(r);
	}
}

/*
 * Generic version, called with pow of 2 threads
 *
 * NOTE: current setup always had 2/3*ninput hidden nodes or greater
 * This means that each thread corresponding to a hidden node can load 2 input nodes into shared mem
 *
 * If doing
 */
__global__ void feed_forward_layer_v1_3(int n_layer1, int n_layer2, float* layer1, float* layer2, float* weights) {
	extern __shared__ float slayer1[]; //get blocksize number of floats

	unsigned int n = blockIdx.x * blockDim.x + threadIdx.x; // node to compute;
	unsigned int tid = threadIdx.x;

	if (tid < n_layer1) // load blocksize node vals into smem, just take the first ones
		slayer1[tid] = layer1[tid];

	__syncthreads();

	if (n < n_layer2) {
		float r = 0;
		for (int i = 0; i <= n_layer1; ++i) { //include bias
			if (i < n_layer1 && i < blockDim.x)
				r += slayer1[i] * weights[n_layer2*i + n];
			else
				r += layer1[i] * weights[n_layer2*i + n];
		}
		layer2[n] = sigmoid(r);
	}
}


__global__ void feed_forward_layer_v2_2(unsigned int pow2, int n_layer1, int n_layer2, float* layer1, float* layer2, float* weights, float* sums) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; // input node
	//printf("x = %d\n", x);
	if (x < (n_layer1+1)*n_layer2) {
		//printf("x = %d\n", x);
		int i = x % (n_layer1+1);
		int j = i % n_layer2;
		int p = j*pow2 + i;
		sums[p] = layer1[i] * weights[n_layer2*i + j];
	}
}

/*
 * generic version
 */
__global__ void compute_activation_v2(float* nodes, float *sums, int n_layer, int stride) {
	unsigned int i = blockIdx.x * blockDim.x+threadIdx.x; // input node

	if (i < n_layer)
		nodes[i] = sigmoid(sums[i*stride]);
}

__global__ void clamp_outputs(float *output, int n) {
	unsigned int i = blockIdx.x * blockDim.x+threadIdx.x;
	if (i < n) {
		output[i] = clamp(output[i]);
	}
}

/*
 * Copied form NVIDIA presentation
 * http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
 */
template <unsigned int blockSize>
__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n, int offset) {
	g_idata = &(g_idata[n*offset]);

	__syncthreads();
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata[tid] = 0;
	while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
	__syncthreads();
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) {if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) {if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32) { if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
		if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
		if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
		if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
		if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
		if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
	}
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


/*
 *
 *
 * ------------ backprop kernels ---------
 *
 *
 */


/*
 * called generically, pow of 2 threads
 */
__global__ void output_error_gradients_v2(float* output, float* target, float* output_err_gradients, int no) {
	unsigned int i = blockIdx.x * blockDim.x+threadIdx.x;

	if (i < no) {
		output_err_gradients[i] = calc_output_gradient(output[i], target[i]);
		//printf("out_err_grad[%d] = %f, output = %f, target = %f\n", i, output_err_gradients[i], output[i], target[i]);
	}
}


/*
 * called generically with power of 2 threads
 */
__global__ void update_hidden_output_deltas_v2(int nh, int no, float l_rate, float momentum,
		float* hidden, float* output_err_gradients, float* delta_ho) {

	unsigned int x = blockIdx.x * blockDim.x+threadIdx.x;

	if (x < (nh+1)*no) { // if in range
		//NOTE: this was my bug, had (x % nh) not (x % (nh+1))
		int j = x % (nh+1); //input node
		int k = x % no; //hidden node

		delta_ho[no*j + k] = l_rate * hidden[j] * output_err_gradients[k] + momentum * delta_ho[no*j + k];
		//printf("delta_ho(%d, %d) = %f, l_rate = %f, hidden[%d] = %f, out_err_gradients[%d] = %f, momentum = %f\n",
		//			j, k, delta_ho[no*j+k], l_rate, j, hidden[j], k, output_err_gradients[k], momentum);
	}
}


__device__ float calc_hidden_gradient(int j, int no, float* hidden, float* d_ho_weights, float* output_err_gradients) {
	//get sum of hidden->output weights * output error gradients
	float s = 0;
	for (int k = 0; k < no; ++k)
		s += d_ho_weights[j*no + k] * output_err_gradients[k];

	//return error gradient
	return hidden[j] * (1 - hidden[j]) * s;
}

/*
 * called generically, pow of 2 threads
 */
__global__ void hidden_error_gradients_v2(int nh, int no, float* hidden, float* d_ho_weights, float* hidden_err_gradients, float* output_err_gradients) {
	unsigned int j = blockIdx.x * blockDim.x+threadIdx.x;

	if (j < nh) { //NOTE: another bug, had (j < (nh+1)*no), only nh nodes need calculated
		hidden_err_gradients[j] = calc_hidden_gradient(j, no, hidden, d_ho_weights, output_err_gradients);
		//printf("hidden_err_grad[%d] = %f\n", j, hidden_err_gradients[j]);
	}
}

/*
 * called with any number of blocks / threads
 * normally, 128 or other power of 2
 */
//TODO: perhaps there is a way to store the hidden_err_gradient[j] in shared memory
__global__ void update_input_hidden_deltas_v2(int ni, int nh, float l_rate, float momentum,
		float* input, float* hidden_err_gradients, float* delta_ih) {
	unsigned int x = blockIdx.x * blockDim.x+threadIdx.x;

	if (x < (ni+1)*nh) {
		int i = x % (ni+1); //input node, NOTE: same bug as before
		int j = x % nh; //hidden node

		delta_ih[nh*i + j] = l_rate * input[i] * hidden_err_gradients[j] + momentum * delta_ih[nh*i + j];

		//printf("delta_ih(%d, %d) = %f, l_rate = %f, input[%d] = %f, hidden_err_gradients[%d] = %f, momentum = %f\n",
		//			i, j, delta_ih[nh*i + j], l_rate, i, input[i], j, hidden_err_gradients[j], momentum);
	}
}


/*
 * called generically with power of 2 threads
 */
__global__ void update_weights_v2(int n1, int n2, float *d_weights, float *deltas, bool reset) {
	unsigned int x = blockIdx.x * blockDim.x+threadIdx.x;

	if (x < (n1+1)*n2) {
		int i = x % (n1+1); //layer 1 node, NOTE: same bug
		int j = x % n2; //layer 2 node

		d_weights[i*n2 + j] += deltas[i*n2 + j];
		//printf("d_weights(%d, %d) = %f, deltas(%d, %d) = %f\n", i, j, d_weights[n2*i+j], i, j, deltas[n2*i + j]);

		if (reset)
			deltas[i*n2 + j] = 0;
	}
}


/*
 *
 * ------- RProp Kernels -----------
 *
 */

/*
 * called generically, pow of 2 threads
 */
__global__ void output_error_gradients_rprop(float* output, float* target, float* output_err_gradients, float* output_err_gradients_tmp, int no) {
	unsigned int i = blockIdx.x * blockDim.x+threadIdx.x;

	if (i < no) {
		output_err_gradients_tmp[i] = output_err_gradients[i];
		output_err_gradients[i] = calc_output_gradient(output[i], target[i]);
		//printf("out_err_grad[%d] = %f, output = %f, target = %f\n", i, output_err_gradients[i], output[i], target[i]);
	}
}

__global__ void update_hidden_output_deltas_rprop(int nh, int no, float step_p, float step_m, float d_max, float d_min,
		float* hidden, float* output_err_gradients, float* output_err_gradients_tmp, float* delta_ho) {

	unsigned int x = blockIdx.x * blockDim.x+threadIdx.x;

	if (x < (nh+1)*no) { // if in range
		int j = x % (nh+1); //input node
		int k = x % no; //hidden node

		int r = output_err_gradients[x] * output_err_gradients_tmp[x];
		if (r > 0) {
			delta_ho[no*j + k] = min(delta_ho[no*j + k] * step_p, d_max);
		} else if (r < 0) {
			delta_ho[no*j + k] = max(delta_ho[no*j + k] * step_m, d_min);
		} else {
			//TODO: need something here for start when delta = 0
		}
	}
}

__global__ void update_weights_rprop(int n1, int n2, float *d_weights, float* gradients, float *deltas) {
	unsigned int x = blockIdx.x * blockDim.x+threadIdx.x;

	if (x < (n1+1)*n2) {
		int i = x % (n1+1); //layer 1 node, NOTE: same bug
		int j = x % n2; //layer 2 node

		int sign = (gradients[j] > 0) - (gradients[j] < 0);
		d_weights[i*n2 + j] = d_weights[i*n2 + j] - sign*deltas[i*n2 + j];
	}
}


/*
 *
 * --------- Debugging ------------
 *
 */

__global__ void print_gpu_net(int n_input, int n_hidden, int n_output,
		float *input, float *hidden, float *output, float *ih_weights, float *ho_weights) {
	for (int i = 0; i <= n_input; ++i) {
		printf("input %d: %f, ", i, input[i]);
	}
	printf("\n");
	for (int i = 0; i <= n_input; ++i) {
		for (int j = 0; j < n_hidden; ++j) {
			printf("ih weight (%d, %d): %f, ", i, j, ih_weights[n_hidden*i + j]);
		}
	}
	printf("\n");
	for (int i = 0; i <= n_hidden; ++i) {
		printf("hidden %d: %f, ", i, hidden[i]);
	}
	printf("\n");
	for (int i = 0; i <= n_hidden; ++i) {
		for (int j = 0; j < n_output; ++j) {
			printf("ho weight (%d, %d): %f, ", i, j, ho_weights[n_output*i + j]);
		}
	}
	printf("\n");
	for (int i = 0; i < n_output; ++i) {
		printf("output %d: %f, ", i, output[i]);
	}
	printf("\n");
}


__global__ void print_target(int n_output, float *target) {
	for (int i = 0; i < n_output; ++i) {
		printf("target[%d] = %f\n", i, target[i]);
	}
}

__global__ void print_input(int n_input, float *input) {
	for (int i = 0; i < n_input+1; i++) {
		printf("input[%d] = %f\n", i, input[i]);
	}
}

/*
 * ---------- Constructors -------------
 */

GPUNet::GPUNet(unsigned int ni, unsigned int no, GPUNetSettings::NetworkStructure net_type=GPUNetSettings::STANDARD) {
	n_input = 0;
	GPUNet::init_structure(ni, no, net_type);
	GPUNet::init_vars();
}

GPUNet::GPUNet(std::string net_file) {
	std::cout << "Initializing from net file: " << net_file << "." << std::endl;
	init_vars();
	read_net(net_file);

	std::cout << "init mse=" << trainingSetMSE << std::endl;
}

GPUNet::~GPUNet() {
	cudaFree(d_input);
	cudaFree(d_hidden);
	cudaFree(d_output);
	cudaFree(d_target);
	cudaFree(d_ih_weights);
	cudaFree(d_ho_weights);
	cudaFree(d_ih_deltas);
	cudaFree(d_ho_deltas);
	cudaFree(d_hid_err_gradients);
	cudaFree(d_out_err_gradients);

	/*
	//I'm getting a bad resource handle error at this line
	CUDA_CHECK_RETURN(cudaStreamDestroy(err_calc_stream));
	CUDA_CHECK_RETURN(cudaStreamDestroy(weight_update_stream1));
	CUDA_CHECK_RETURN(cudaStreamDestroy(weight_update_stream2));
	CUDA_CHECK_RETURN(cudaStreamDestroy(train_stream1));
	CUDA_CHECK_RETURN(cudaStreamDestroy(train_stream2));
	CUDA_CHECK_RETURN(cudaEventDestroy(event1));
	CUDA_CHECK_RETURN(cudaEventDestroy(event2));*/

	delete[] h_output;
	delete[] h_ih_weights;
	delete[] h_ho_weights;
	delete[] gpu_mem;
	delete nio;
}

/*
 * -------------- public ---------------
 */


void GPUNet::init_structure(unsigned int ni, unsigned int no, GPUNetSettings::NetworkStructure net_type) {
	if (n_input != 0) { // constructor initializing nodes has been called, error out
		std::cerr << "Network has already been initialized" << std::endl;
	} else if (ni != 0) { // if not empty constructor
		n_input = ni;
		n_output = no;
		GPUNet::net_type = net_type;
		if (net_type == GPUNetSettings::STANDARD) {
			n_hidden = ceil(2.0/3.0*ni);
		} else if (net_type == GPUNetSettings::GPU_ARCH_OPT) {
			//get first multiple of 128 greater than 2.0/3.0*ni
			n_hidden = (2.0/3.0*ni+127) / 128 * 128;
		} else {
			std::cerr << "Invalid network type: " << net_type << std::endl;
			exit(1);
		}
	}
}

void GPUNet::init_vars() {
	nio = new NetIO();
	nio->set_gnet(this);

	max_epochs = GPUNetSettings::GPU_MAX_EPOCHS;
	l_rate = GPUNetSettings::GPU_LEARNING_RATE;
	momentum = GPUNetSettings::GPU_MOMENTUM;
	desired_acc = GPUNetSettings::GPU_DESIRED_ACCURACY;
	batching = GPUNetSettings::GPU_USE_BATCH;
	save_freq = GPUNetSettings::GPU_SAVE_FREQUENCY;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&n_gpus));

	epoch = 0;
	trainingSetAccuracy = 0;
	validationSetAccuracy = 0;
	generalizationSetAccuracy = 0;
	trainingSetMSE = 0;
	validationSetMSE = 0;
	generalizationSetMSE = 0;

	start = 0;
	finish = 0;

	/*
	 * device
	 */
	d_input = NULL;
	d_hidden = NULL;
	d_output = NULL;
	d_target = NULL;

	d_ih_weights = NULL;
	d_ho_weights = NULL;

	d_ih_deltas = NULL;
	d_ho_deltas = NULL;

	d_hid_err_gradients = NULL;
	d_out_err_gradients = NULL;

	CUDA_CHECK_RETURN(cudaStreamCreate(&err_calc_stream));
	CUDA_CHECK_RETURN(cudaStreamCreate(&weight_update_stream1));
	CUDA_CHECK_RETURN(cudaStreamCreate(&weight_update_stream2));
	CUDA_CHECK_RETURN(cudaStreamCreate(&train_stream1));
	CUDA_CHECK_RETURN(cudaStreamCreate(&train_stream2));
	CUDA_CHECK_RETURN(cudaEventCreate(&event1));
	CUDA_CHECK_RETURN(cudaEventCreate(&event2));

	/*
	 * host validation
	 */
	h_output = new float[n_output];
	h_ih_weights = new float[(n_input+1)*n_hidden];
	h_ho_weights = new float[(n_hidden+1)*n_output];

	//init gpu mem to 0 for each gpu
	gpu_mem = new size_t[n_gpus];
	memset(gpu_mem, 0, n_gpus*sizeof(size_t));
}


/*
 * allocate memory on device for
 * input, hidden, output, target
 * ih_weights, ho_weights
 * ih_deltas, ho_deltas
 * hid_err_gradients
 * out_err_gradients
 */

void GPUNet::alloc_dev_mem() {
	//nodes
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_input, (n_input+1)*sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_hidden, (n_hidden+1)*sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_output, (n_output)*sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_target, (n_output)*sizeof(float)));
	add_gpu_mem((n_input+n_hidden+(2*n_output)+2)*sizeof(float));

	//weights
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_ih_weights, ((n_input+1)*n_hidden)*sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_ho_weights, ((n_hidden+1)*n_output)*sizeof(float)));
	add_gpu_mem(((n_input+1)*n_hidden + (n_hidden+1)*n_output)*sizeof(float));

	//create delta arrays, include bias
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_ih_deltas, ((n_input+1)*n_hidden)*sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_ho_deltas, ((n_hidden+1)*n_output)*sizeof(float)));
	add_gpu_mem(((n_input+1)*n_hidden + (n_hidden+1)*n_output)*sizeof(float));

	//error gradients
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_hid_err_gradients, (n_hidden+1)*sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_out_err_gradients, (n_output+1)*sizeof(float)));
	add_gpu_mem((n_hidden + n_output + 2)*sizeof(float));

	std::cout << "Memory allocated on device" << std::endl;
}

/*
 * Note: assumes sizes of networks are the same
 * This is for testing purposes so that
 * I can have identical networks.
 */
void GPUNet::init_from_net(Net &net, NetData &d) {
	int threads = 128;

	//copy first pattern to input neurons so it is copied to device, instead of zeros
	for (int i = 0; i < net.n_input; ++i) {
		net.inputNeurons[i] = d.get_training_dataset()->training_set[0]->input[i];
	}

	// so hidden and output initialized to 0
	CUDA_CHECK_RETURN(cudaMemcpy(d_input, net.inputNeurons, (net.n_input)*sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_hidden, net.hiddenNeurons, (net.n_hidden)*sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_output, net.outputNeurons, (net.n_output)*sizeof(float), cudaMemcpyHostToDevice));

	set_bias<<<1,1>>>(n_input, d_input);
	set_bias<<<1,1>>>(n_hidden, d_hidden);

	CUDA_CHECK_RETURN(cudaMemcpy(d_ih_weights, net.wInputHidden, (net.n_input+1)*(net.n_hidden)*sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_ho_weights, net.wHiddenOutput, (net.n_hidden+1)*(net.n_output)*sizeof(float), cudaMemcpyHostToDevice));

	init_deltas_v2<<<((n_input+1)*n_hidden+threads-1)/threads, threads>>>(n_input+1, n_hidden, d_ih_deltas);
	init_deltas_v2<<<((n_hidden+1)*n_output+threads-1)/threads, threads>>>(n_hidden+1, n_output, d_ho_deltas);

	std::cout << "Data copied to device" << std::endl << std::endl;
}


void GPUNet::init_net() {
	int threads = 128;

	//init nodes to all 0
	init_nodes_layer_v2<<<(n_input+1+threads-1)/threads, threads>>>(n_input+1, d_input);
	init_nodes_layer_v2<<<(n_hidden+1+threads-1)/threads, threads>>>(n_hidden+1, d_hidden);
	init_nodes_output_v2<<<(n_output+threads-1)/threads, threads>>>(n_output, d_output);

	//init weights to random vals
	curandState *state;
	CUDA_CHECK_RETURN(cudaMalloc(&state, threads*sizeof(curandState)));
	curand_setup<<<1, threads>>>(state);

	init_weights_v2<<<((n_input+1)*n_hidden+threads-1)/threads, threads>>>(n_input+1, n_hidden, d_ih_weights, state);
	init_weights_v2<<<((n_hidden+1)*n_output+threads-1)/threads, threads>>>(n_hidden+1, n_output, d_ho_weights, state);

	CUDA_CHECK_RETURN(cudaFree(state));

	//init deltas to 0
	init_deltas_v2<<<((n_input+1)*n_hidden+threads-1)/threads, threads>>>(n_input+1, n_hidden, d_ih_deltas);
	init_deltas_v2<<<((n_hidden+1)*n_output+threads-1)/threads, threads>>>(n_hidden+1, n_output, d_ho_deltas);

	CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

void GPUNet::set_learning_rate(float lr) {
	l_rate = lr;
}

void GPUNet::set_momentum(float m) {
	momentum = m;
}

void GPUNet::set_training_params(float lr, float m, bool b) {
	l_rate = lr;
	momentum = m;
	batching = b;
}

void GPUNet::set_max_epochs(int me) {
	max_epochs = me;
}

void GPUNet::set_save_frequency(int f) {
	save_freq = f;
}

void GPUNet::set_desired_accuracy(float acc) {
	desired_acc = acc;
}

void GPUNet::set_stopping_conds(int me, float acc) {
	max_epochs = me;
	desired_acc = acc;
}


/*
 * to keep it simple, run in 1 thread
 */
void GPUNet::print_net() {
	print_gpu_net<<<1, 1>>>(n_input, n_hidden, n_output,
			d_input, d_hidden, d_output, d_ih_weights, d_ho_weights);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}


/*
 * transfer weights back to host
 * write important data (num_epochs, layers, nodes/layer, l_rate, momentum, max_epochs, desired_acc, current mse, current acc)
 *
 */
bool GPUNet::write_net(std::string fname) {
	//need to copy mse and acc back to host
	copy_error_to_host();
	std::cout << "current acc=" << trainingSetAccuracy << ", current mse=" << trainingSetMSE << std::endl;

	if (!nio->write_net(fname)) {
		std::cerr << "Write failed" << std::endl;
		return false;
	}
	return true;
}

bool GPUNet::read_net(std::string fname) {
	if (!nio->read_net(fname)) {
		std::cerr << "Read failed" << std::endl;
		return false;
	}

	int threads = 128;
	//init nodes to 0
	init_nodes_layer_v2<<<(n_input+1+threads-1)/threads, threads>>>(n_input+1, d_input);
	init_nodes_layer_v2<<<(n_hidden+1+threads-1)/threads, threads>>>(n_hidden+1, d_hidden);
	init_nodes_output_v2<<<(n_output+threads-1)/threads, threads>>>(n_output, d_output);

	//init deltas to 0
	init_deltas_v2<<<((n_input+1)*n_hidden+threads-1)/threads, threads>>>(n_input+1, n_hidden, d_ih_deltas);
	init_deltas_v2<<<((n_hidden+1)*n_output+threads-1)/threads, threads>>>(n_hidden+1, n_output, d_ho_deltas);

	return true;
}

/*
 * run the input through the network
 */
float* GPUNet::evaluate(float* input) {
	//copy to device
	//feed forward
	//copy back output
	int threads = 128;
	float *h_out = new float[n_output];
	CUDA_CHECK_RETURN(cudaMemcpy((void*)d_input, (void*)input, n_input*sizeof(float), cudaMemcpyHostToDevice));
	feed_forward_v1_2(d_input);
	clamp_outputs<<<(n_output+threads-1)/threads, threads>>>(d_output, n_output);
	CUDA_CHECK_RETURN(cudaMemcpy(h_out, d_output, n_output*sizeof(float), cudaMemcpyDeviceToHost));
	return h_out;
}

float* GPUNet::batch_evaluate(float** inputs) {
	//copy to device
	//feed forward
	//copy back output
	int threads = 128;
	float *h_out = new float[n_output];
	CUDA_CHECK_RETURN(cudaMemcpy((void*)d_input, (void*)inputs, n_input*sizeof(float), cudaMemcpyHostToDevice));
	feed_forward_v1_2(d_input);
	clamp_outputs<<<(n_output+threads-1)/threads, threads>>>(d_output, n_output);
	CUDA_CHECK_RETURN(cudaMemcpy(h_out, d_output, n_output*sizeof(float), cudaMemcpyDeviceToHost));
	return h_out;
}

int GPUNet::get_num_input() {
	return n_input;
}

int GPUNet::get_num_hidden() {
	return n_hidden;
}

int GPUNet::get_num_output() {
	return n_output;
}


void GPUNet::calc_dataset_parameters(TrainingDataSet *tset) {
	std::cout << "Determining data set statistics" << std::endl;
	// calc num patterns copyable
	// num patterns = integer div of available memory / mem for single pattern
	int bytes_per_pattern = sizeof(float)*((n_input+1)+(n_output));
	int cur_dev = get_current_device();
	std::cout << " bytes per pattern = "<<bytes_per_pattern<<std::endl;
	std::cout << " total dev mem = "<< total_dev_mem(cur_dev)<<std::endl;
	std::cout << " current mem usage = "<< current_mem_usage(cur_dev)<<std::endl;
	int available_mem = total_dev_mem(cur_dev) - current_mem_usage(cur_dev);
	std::cout << " available mem = "<<available_mem<<std::endl;
	std::cout << " tset.size = "<<tset->size()<<std::endl;
	n_copyable_patterns = available_mem / bytes_per_pattern;

	//ensure n_copyable_patterns is even and can be split into 2 buffers
	if (n_copyable_patterns % 2 == 1) {
		--n_copyable_patterns;
	}

	if (n_copyable_patterns > tset->size()) {
		n_copyable_patterns = tset->size();
	}
	// calc num sections
	// num_sections = ceil ( n_patterns / n_copyable_patterns)
	n_sections = (tset->size() + n_copyable_patterns - 1) / n_copyable_patterns;
	std::cout << " n_copyable_patterns = "<<n_copyable_patterns<<", n_sections = "<<n_sections<<std::endl<<std::endl;
}

void GPUNet::train_net_sectioned_overlap(TrainingDataSet *tset) {
	calc_dataset_parameters(tset);

	std::cout << std::endl << "Neural Network Training Starting: " << std::endl
			<< "----------------------------------------------------" << std::endl
			<< "LR: " << l_rate << ", Momentum: " << momentum << ", Max Epochs: " << max_epochs << std::endl
			<< n_input << " Input Neurons, " << n_hidden << " Hidden Neurons, " << n_output << " Output Neurons" << std::endl
			<< "----------------------------------------------------" << std::endl << std::endl;

	int buffer_size = n_copyable_patterns / 2;
	FeatureVector** d_training_set1;
	FeatureVector** d_training_set2;

	copy_to_device_host_array_ptrs_biased_section_stream(tset->training_set, &d_training_set1, 0, buffer_size, true, train_stream1);
	CUDA_CHECK_RETURN(cudaEventRecord(event1, train_stream1));
	CUDA_CHECK_RETURN(cudaStreamWaitEvent(train_stream2, event1, 0));

	while (epoch < max_epochs) {
		std::cout << "Epoch: " << epoch << std::endl;
		//copy a section and run partial epoch
		for (int i = 0; i < n_sections*2; ++i) {
			//copy patterns from [n_sections*n_patterns_copyable, (n_sections+1)*n_patterns_copyable)
			int p_start = (i+1) * buffer_size;
			int p_end = p_start + buffer_size;
			if (p_end > tset->training_set.size()) p_end = tset->training_set.size();

			std::cout << "copying section="<<i<<", pstart="<< p_start << ", pend="<<p_end << std::endl;
			if (i % 2 == 0) {
				// use buffer 1
				// copy to buffer 2
				copy_to_device_host_array_ptrs_biased_section_stream(tset->training_set, &d_training_set2, p_start, p_end, false, train_stream1);
				run_training_epoch_dev(d_training_set1, p_end-p_start);
			} else { // i % 2 == 1
				// use buffer 2
				// copy to buffer 1
				copy_to_device_host_array_ptrs_biased_section_stream(tset->training_set, &d_training_set1, p_start, p_end, false, train_stream1);
				run_training_epoch_dev(d_training_set2, p_end-p_start);
			}
		}

		//once training set is complete increment epoch
		++epoch;
	}

	//out validation accuracy and MSE
	std::cout << std::endl << "Training complete. Elapsed epochs: " << epoch << std::endl;

	copy_error_to_host();
	std::cout << "MSE = " << trainingSetMSE << std::endl;
	std::cout << "ACC = " << trainingSetAccuracy << std::endl;

	//free training set
	for (int i = 0; i < tset->training_set.size(); ++i) {
		CUDA_CHECK_RETURN(cudaFree(d_training_set1[i]->input));
		CUDA_CHECK_RETURN(cudaFree(d_training_set1[i]->target));
		free(d_training_set1[i]);
		CUDA_CHECK_RETURN(cudaFree(d_training_set2[i]->input));
		CUDA_CHECK_RETURN(cudaFree(d_training_set2[i]->target));
		free(d_training_set2[i]);

	}
	free(d_training_set1);
	free(d_training_set2);
}


void GPUNet::train_net_sectioned(TrainingDataSet *tset) {
	calc_dataset_parameters(tset);

	std::cout << std::endl << "Neural Network Training Starting: " << std::endl
			<< "----------------------------------------------------" << std::endl
			<< "LR: " << l_rate << ", Momentum: " << momentum << ", Max Epochs: " << max_epochs << std::endl
			<< n_input << " Input Neurons, " << n_hidden << " Hidden Neurons, " << n_output << " Output Neurons" << std::endl
			<< "----------------------------------------------------" << std::endl << std::endl;

	FeatureVector** d_training_set;

	if (n_sections == 1) { // no section copying necessary
		copy_to_device_host_array_ptrs_biased(tset->training_set, &d_training_set);
		while (epoch < max_epochs) {
			std::cout << "Epoch: " << epoch << ", ";
			run_training_epoch_dev(d_training_set, tset->training_set.size());
			++epoch;

			if (epoch % save_freq == 0) {
				std::cout << "starting save" << std::endl;
				std::string fname = "nets/face_" + boost::lexical_cast<std::string>(epoch) + ".net";
				std::cout << "Writing intermediary net " << fname << std::endl;
				write_net(fname);
				std::cout << "finished save" << std::endl;
			}
		}
	} else {
		while (epoch < max_epochs) {
			std::cout << "Epoch: " << epoch << std::endl;
			//copy a section and run partial epoch
			for (int i = 0; i < n_sections; ++i) {
				//copy patterns from [n_sections*n_patterns_copyable, (n_sections+1)*n_patterns_copyable)
				int p_start = i * n_copyable_patterns;
				int p_end = p_start + n_copyable_patterns;
				if (p_end > tset->training_set.size()) p_end = tset->training_set.size();
				std::cout << "copying section="<<i<<", pstart="<< p_start << ", pend="<<p_end << std::endl;
				copy_to_device_host_array_ptrs_biased_section(tset->training_set, &d_training_set, p_start, p_end, i == 0 && epoch == 0);
				std::cout << "data copied" << std::endl;
				run_training_epoch_dev(d_training_set, p_end-p_start);
			}

			//once training set is complete increment epoch
			++epoch;
		}
	}

	//out validation accuracy and MSE
	std::cout << std::endl << "Training complete. Elapsed epochs: " << epoch << std::endl;

	copy_error_to_host();
	std::cout << "MSE = " << trainingSetMSE << std::endl;
	std::cout << "ACC = " << trainingSetAccuracy << std::endl;

	//free training set
	for (int i = 0; i < tset->training_set.size(); ++i) {
		CUDA_CHECK_RETURN(cudaFree(d_training_set[i]->input));
		CUDA_CHECK_RETURN(cudaFree(d_training_set[i]->target));
		free(d_training_set[i]);
	}
	free(d_training_set);
}

void GPUNet::copy_error_to_host() {
	CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&trainingSetMSE, d_mse, sizeof(float), 0, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&trainingSetAccuracy, d_acc, sizeof(float), 0, cudaMemcpyDeviceToHost));
}


void GPUNet::run_training_epoch_dev(FeatureVector **feature_vecs, size_t n_features) {
	int n_threads = 128;
	start = clock();
	for (size_t i = 0; i < n_features; ++i) {
		feed_forward_v1_2(feature_vecs[i]->input);
		backprop_v2(feature_vecs[i]->input, feature_vecs[i]->target);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	}
	if (batching) { //update weights here and reset deltas
		update_weights_v2<<<((n_output*(n_hidden+1))+n_threads-1)/n_threads, n_threads, 0, weight_update_stream1>>>(n_hidden, n_output, d_ho_weights, d_ho_deltas, true);
		update_weights_v2<<<((n_hidden*(n_input+1))+n_threads-1)/n_threads, n_threads, 0, weight_update_stream2>>>(n_input, n_hidden, d_ih_weights, d_ih_deltas, true);
	}
	calc_mse<<<1, 1>>>(n_output, n_features);
	calc_acc<<<1, 1>>>(n_features);
	finish = clock();
	std::cout << "time: " << ((double)finish-start)/CLOCKS_PER_SEC << std::endl;
}



void GPUNet::backprop_v2(float *d_inp, float *d_tar) {
	int n_threads = 128;

	//maintain mse state
	mse_sum_v2<<<1, 1, 0, err_calc_stream>>>(d_output, d_tar, n_output);
	output_correct_v2<<<1, 1, 0, err_calc_stream>>>(d_output, d_tar, n_output);
	//CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	//float mse_sum = 0;
	//CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&mse_sum, d_mse_sum, sizeof(float), 0, cudaMemcpyDeviceToHost));
	//std::cout << "Current mse_sum = " << mse_sum << std::endl;

	output_error_gradients_v2<<<(n_output+n_threads-1)/n_threads, n_threads, 0, 0>>>(d_output, d_tar, d_out_err_gradients, n_output);
	//CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	update_hidden_output_deltas_v2<<<((n_output*(n_hidden+1))+n_threads-1)/n_threads, n_threads, 0, 0>>>(n_hidden, n_output, l_rate, momentum, d_hidden, d_out_err_gradients, d_ho_deltas);
	//CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	hidden_error_gradients_v2<<<(n_hidden+n_threads-1)/n_threads, n_threads, 0, 0>>>(n_hidden, n_output, d_hidden, d_ho_weights,
			d_hid_err_gradients, d_out_err_gradients);
	//CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	if (!batching) { // don't update weights here
		CUDA_CHECK_RETURN(cudaEventRecord(event1));
		CUDA_CHECK_RETURN(cudaStreamWaitEvent(weight_update_stream1, event1, 0));
		update_weights_v2<<<((n_output*(n_hidden+1))+n_threads-1)/n_threads, n_threads, 0, weight_update_stream1>>>(n_hidden, n_output, d_ho_weights, d_ho_deltas, false);
	}

	update_input_hidden_deltas_v2<<<((n_hidden*(n_input+1))+n_threads-1)/n_threads, n_threads, 0, 0>>>(n_input, n_hidden, l_rate, momentum,
			d_inp, d_hid_err_gradients, d_ih_deltas);

	if (!batching) { // don't update weights here
		CUDA_CHECK_RETURN(cudaEventRecord(event2));
		CUDA_CHECK_RETURN(cudaStreamWaitEvent(weight_update_stream2, event2, 0));

//		int n_streams = 4;
//		cudaStream_t *streams = (cudaStream_t*)malloc(n_streams*sizeof(cudaStream_t));
//
//		for (int i = 0; i < n_streams; ++i) {
//			CUDA_CHECK_RETURN(cudaStreamCreate(&streams[i]));
//			update_weights_v2<<<((n_hidden*(n_input+1))+n_threads-1)/n_threads, n_threads, 0, streams[i]>>>(n_input, n_hidden, d_ih_weights, d_ih_deltas, false);
//		}
//
		update_weights_v2<<<((n_hidden*(n_input+1))+n_threads-1)/n_threads, n_threads, 0, weight_update_stream2>>>(n_input, n_hidden, d_ih_weights, d_ih_deltas, false);
//		for (int i = 0; i < n_streams; ++i) {
//			CUDA_CHECK_RETURN(cudaStreamDestroy(streams[i]));
//		}
	}
}


void GPUNet::rprop(float *d_inp, float *d_tar) {
	int n_threads = 128;
	//calc hidden out gradients
	//

}


void GPUNet::feed_forward_v1_2(float *d_inp) {
	int threads = 128;
	feed_forward_layer_v1_2<<<(n_hidden+threads-1)/threads, threads>>>(n_input, n_hidden, d_inp, d_hidden, d_ih_weights);
	feed_forward_layer_v1_2<<<(n_output+threads-1)/threads, threads>>>(n_hidden, n_output, d_hidden, d_output, d_ho_weights);
}

void GPUNet::feed_forward_v1_3(float *d_inp) {
	int threads = 128;
	feed_forward_layer_v1_3<<<(n_hidden+threads-1)/threads, threads, threads*sizeof(float)>>>(n_input, n_hidden, d_inp, d_hidden, d_ih_weights);
	feed_forward_layer_v1_3<<<(n_output+threads-1)/threads, threads, threads*sizeof(float)>>>(n_hidden, n_output, d_hidden, d_output, d_ho_weights);
}


void GPUNet::feed_forward_v2_2(unsigned int pow2, float *d_inp, float *d_sums) {
	int threads = 128;
	feed_forward_layer_v2_2<<<((n_input+1)*n_hidden+threads-1)/threads, threads>>>(pow2, n_input, n_hidden, d_inp, d_hidden, d_ih_weights, d_sums);


	float *h_sums, *h_tmp;
	h_sums = (float*)malloc(pow2*(n_hidden)*sizeof(float));
	memset(h_sums, 0, pow2*(n_hidden)*sizeof(float));
	h_tmp = (float*)malloc(pow2*sizeof(float));
	memset(h_tmp, 0, pow2*sizeof(float));

	CUDA_CHECK_RETURN(cudaMemcpy(h_sums, d_sums, pow2*(n_hidden)*sizeof(float), cudaMemcpyDeviceToHost));
	for (int i = 0; i < pow2*(n_hidden); ++i) {
		std::cout << h_sums[i] << " ";
	}
	std::cout << std::endl;

	float *d_tmp;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_tmp, pow2*sizeof(float)));
	CUDA_CHECK_RETURN(cudaMemset(d_tmp, 0, pow2*sizeof(float)));
	//reduce_kernel<128><<<(pow2+threads-1)/threads, threads, threads*sizeof(float)>>>(d_sums, d_tmp, pow2, 0);

	reduce_kernel<128><<<(pow2+threads-1)/threads, threads, threads*sizeof(float)>>>(d_sums, d_tmp, pow2, 1);

	CUDA_CHECK_RETURN(cudaMemcpy(h_tmp, d_tmp, pow2*sizeof(float), cudaMemcpyDeviceToHost));
	for (int i = 0; i < pow2; ++i) {
		std::cout << h_tmp[i] << " ";
	}
	std::cout << std::endl;

	//compute_activation_v2<<<(n_hidden+threads-1)/n_hidden, threads>>>(d_hidden, d_sums_l1, n_hidden, n_input+1);
}

bool GPUNet::validate_output(float* desired_output) {
	//copy output back to host
	CUDA_CHECK_RETURN(cudaMemcpy(h_output, d_output, n_output*sizeof(float), cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_output; ++i) {
		//std::cout << "actual = " << desired_output[i] << ", calc = " << h_output[i] << std::endl;
		if (abs(desired_output[i] - h_output[i]) > .005)
			return false;
	}
	return true;
}

bool GPUNet::validate_weights(float *desired_ih_weights, float *desired_ho_weights) {
	//copy inp hid weights to host

	CUDA_CHECK_RETURN(cudaMemcpy(h_ih_weights, d_ih_weights, (n_input+1)*n_hidden*sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(h_ho_weights, d_ho_weights, (n_hidden+1)*n_output*sizeof(float), cudaMemcpyDeviceToHost));

	for (int i = 0; i < (n_input+1)*n_hidden; ++i) {
		if (abs(desired_ih_weights[i] - h_ih_weights[i]) > .005)
			return false;
	}

	for (int i = 0; i < (n_hidden+1)*n_output; ++i) {
		if (abs(desired_ho_weights[i] - h_ho_weights[i]) > .005)
			return false;
	}

	return true;
}


void GPUNet::test_feed_forward(Net &net, NetData &d) {
	clock_t start, finish;

	std::cout << "feed forward CPU" << std::endl;
	start = clock();
	net.feed_forward(d.get_training_dataset()->training_set[0]->input);
	finish = clock();
	std::cout << "feed forward CPU time: " << ((float)(finish-start)) / CLOCKS_PER_SEC << "s\n\n";
	//net.print_network();

	FeatureVector **dv;
	GPUNet::copy_to_device_host_array_ptrs_biased(d.get_training_dataset()->training_set, &dv);
	std::cout << "Testing method 1.2" << std::endl;
	feed_forward_v1_2(dv[0]->input);
	std::cout << "Validates: " << validate_output(net.outputNeurons) << "\n";
	//net.print_network();
	//print_net();
	CUDA_CHECK_RETURN(cudaMemset(d_output, 0, n_output*sizeof(float)));

	std::cout << "Testing method 1.3" << std::endl;
	feed_forward_v1_3(dv[0]->input);
	std::cout << "Validates: " << validate_output(net.outputNeurons) << "\n";
	//net.print_network();
	//print_net();
	CUDA_CHECK_RETURN(cudaMemset(d_output, 0, n_output*sizeof(float)));

	/*std::cout << "Testing method 2" << std::endl;
	feed_forward_v2();
	std::cout << "Validates: " << validates(net.outputNeurons) << "\n";
	CUDA_CHECK_RETURN(cudaMemset(d_output, 0, n_output*sizeof(float)));

	std::cout << "Testing method 2.2" << std::endl;
	feed_forward_v2_2();
	std::cout << "Validates: " << validates(net.outputNeurons) << "\n";
	CUDA_CHECK_RETURN(cudaMemset(d_output, 0, n_output*sizeof(float)));*/
}

void GPUNet::test_backprop(Net &net, NetData &d) {
	NetTrainer nt(&net);
	//std::cout << "CPU net 0" << std::endl;
	//net.print_network();

	net.feed_forward(d.get_training_dataset()->training_set[0]->input);
	//std::cout << "CPU net 1" << std::endl;
	//net.print_network();

	nt.backprop(d.get_training_dataset()->training_set[0]->target);
	//std::cout << "CPU net 2" << std::endl;
	//net.print_network();

	std::cout << "Testing backprop_v2" << std::endl;
	FeatureVector **dv;
	GPUNet::copy_to_device_host_array_ptrs_biased(d.get_training_dataset()->training_set, &dv);

	//std::cout << std::endl << "GPU net 0" << std::endl;
	//print_net();
	//std::cout << std::endl;

	feed_forward_v1_2(dv[0]->input);
	//std::cout << "GPU net 1" << std::endl;
	//print_net();
	//std::cout << std::endl;

	//std::cout << "GPU net 2" << std::endl;
	backprop_v2(dv[0]->input, dv[0]->target);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	//print_net();
	//std::cout << std::endl;
	std::cout << "Validates: " << validate_weights(net.wInputHidden, net.wHiddenOutput) << std::endl;

//	net.feed_forward(d.get_training_dataset()->training_set[1]->input);
//	nt.backprop(d.get_training_dataset()->training_set[1]->target);
//	nt.update_weights();
//	feed_forward_v1_2(dv[1]->input);
//	backprop_v2(dv[1]->input, dv[1]->target);
//
//
//	std::cout << "Validates: " << validate_weights(net.wInputHidden, net.wHiddenOutput) << std::endl;
}

void GPUNet::run_parallel(Net &net, NetData &d) {
	std::cout << "Running in parallel" <<std::endl;

	FeatureVector **dv;
	GPUNet::copy_to_device_host_array_ptrs_biased(d.get_training_dataset()->training_set, &dv);

	NetTrainer nt(&net);

	int e = 0;
	std::string r = "";
	while (true) {
		std::cout << "Epoch " << e++ << std::endl;
		for (int i = 0; i < d.get_training_dataset()->training_set.size(); ++i) {
			net.feed_forward(d.get_training_dataset()->training_set[i]->input);
			nt.backprop(d.get_training_dataset()->training_set[i]->target);

			feed_forward_v1_2(dv[0]->input);
			backprop_v2(dv[0]->input, dv[0]->target);

			std::cout << "CPU network" << std::endl;
			net.print_network();
			std::cout << "GPU network" << std::endl;
			print_net();
			std::cout << "Validates: " << validate_weights(net.wInputHidden, net.wHiddenOutput) << std::endl;
			std::getline(std::cin, r);
			if (r == "exit") {
				return;
			}
		}
	}
}



size_t GPUNet::current_mem_usage(int dev) {
	return gpu_mem[dev];
}



/*
 * ------------ private ------------
 */


void GPUNet::add_gpu_mem(int bytes) {
	gpu_mem[get_current_device()] += bytes;
}


int GPUNet::get_current_device() {
	int device;
	cudaGetDevice(&device);
	return device;
}

size_t GPUNet::dataset_size(TrainingDataSet *tset) {
	size_t tset_size = 0;
	int fv_size = (n_input + n_output) * sizeof(float);
	tset_size += fv_size * tset->training_set.size();
	tset_size += fv_size * tset->generalization_set.size();
	tset_size += fv_size * tset->validation_set.size();
	return tset_size;
}

size_t GPUNet::total_dev_mem(int dev) {
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, dev);
	return props.totalGlobalMem; // - 1611000000; //minus 1.5 gb
}



/*
 * Copies the host vector to a pointer array on the host that holds pointers to FeatureVector on the device with bias node
 */
void GPUNet::copy_to_device_host_array_ptrs_biased(thrust::host_vector<FeatureVector*> &hv, FeatureVector ***dv) {
	std::cout << "Copying data" << std::endl;
	start = clock();

	*dv = (FeatureVector**)malloc(hv.size()*sizeof(FeatureVector*));
	//FeatureVector** host_dv_tmp = (FeatureVector**)malloc(hv.size()*sizeof(FeatureVector*));

	for (size_t i = 0; i < hv.size(); ++i) {
		//allocate device memory
		FeatureVector *d_fv = (FeatureVector*)malloc(sizeof(FeatureVector*));

		float *d_inp, *d_tar;
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_inp, (n_input+1)*sizeof(float)));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_tar, (n_output)*sizeof(float)));

		//TODO: cuda-memcheck claims there is an unspecified launch failure at this line...
		CUDA_CHECK_RETURN(cudaMemcpy(d_inp, hv[i]->input, n_input*sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(d_tar, hv[i]->target, n_output*sizeof(float), cudaMemcpyHostToDevice));

		d_fv->input = d_inp;
		d_fv->target = d_tar;

		//TODO: does setting all in parallel improve speed?
		set_bias<<<1, 1>>>(n_input, d_inp);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		(*dv)[i] = d_fv;
	}
	finish = clock();
	std::cout << "Copy time: " << ((float)finish-start)/CLOCKS_PER_SEC << std::endl;
}

/**
 * Copy from pattern p_start to p_end to device
 * only allocate memory if \p allocate is true
 */
void GPUNet::copy_to_device_host_array_ptrs_biased_section(thrust::host_vector<FeatureVector*> &hv, FeatureVector ***dv,
		int p_start, int p_end, bool allocate) {

	std::cout << "Copying data, p_start = " << p_start << ", p_end = " << p_end << ", allocate = " << allocate << std::endl;
	start = clock();

	if (allocate) { // if the first epoch and the first section
		*dv = (FeatureVector**)malloc(hv.size()*sizeof(FeatureVector*));
	}

	for (int i = p_start, p = 0; i < p_end; ++i, ++p) {
		if (allocate) {
			//allocate device memory
			FeatureVector *d_fv = (FeatureVector*)malloc(sizeof(FeatureVector*));

			float *d_inp, *d_tar;
			CUDA_CHECK_RETURN(cudaMalloc((void **)&d_inp, (n_input+1)*sizeof(float)));
			CUDA_CHECK_RETURN(cudaMalloc((void **)&d_tar, (n_output)*sizeof(float)));

			CUDA_CHECK_RETURN(cudaMemcpy(d_inp, hv[i]->input, n_input*sizeof(float), cudaMemcpyHostToDevice));
			CUDA_CHECK_RETURN(cudaMemcpy(d_tar, hv[i]->target, n_output*sizeof(float), cudaMemcpyHostToDevice));

			d_fv->input = d_inp;
			d_fv->target = d_tar;

			//NOTE: no need to synchronize since on default stream and
			//next GPU function could not start until this one finishes
			set_bias<<<1, 1>>>(n_input, d_inp);

			(*dv)[p] = d_fv;
		} else {
			CUDA_CHECK_RETURN(cudaMemcpy((*dv)[p]->input, hv[i]->input, n_input*sizeof(float), cudaMemcpyHostToDevice));
			CUDA_CHECK_RETURN(cudaMemcpy((*dv)[p]->target, hv[i]->target, n_output*sizeof(float), cudaMemcpyHostToDevice));
		}
	}
	finish = clock();
	std::cout << "Copy time: " << ((float)finish-start)/CLOCKS_PER_SEC << std::endl;
}


/**
 * Copy from pattern p_start to p_end to device
 * only allocate memory if \p allocate is true
 */
void GPUNet::copy_to_device_host_array_ptrs_biased_section_stream(thrust::host_vector<FeatureVector*> &hv, FeatureVector ***dv,
		int p_start, int p_end, bool allocate, cudaStream_t stream) {

	std::cout << "Copying data, p_start = " << p_start << ", p_end = " << p_end << ", allocate = " << allocate << std::endl;
	start = clock();

	if (allocate) { // if the first epoch and the first section
		*dv = (FeatureVector**)malloc(hv.size()*sizeof(FeatureVector*));
	}

	for (int i = p_start, p = 0; i < p_end; ++i, ++p) {
		if (allocate) {
			//allocate device memory
			FeatureVector *d_fv = (FeatureVector*)malloc(sizeof(FeatureVector*));

			float *d_inp, *d_tar;
			CUDA_CHECK_RETURN(cudaMalloc((void **)&d_inp, (n_input+1)*sizeof(float)));
			CUDA_CHECK_RETURN(cudaMalloc((void **)&d_tar, (n_output)*sizeof(float)));

			CUDA_CHECK_RETURN(cudaMemcpyAsync(d_inp, hv[i]->input, n_input*sizeof(float), cudaMemcpyHostToDevice, stream));
			CUDA_CHECK_RETURN(cudaMemcpyAsync(d_tar, hv[i]->target, n_output*sizeof(float), cudaMemcpyHostToDevice, stream));

			d_fv->input = d_inp;
			d_fv->target = d_tar;

			//NOTE: no need to synchronize since on default stream and
			//next GPU function could not start until this one finishes
			set_bias<<<1, 1>>>(n_input, d_inp);

			(*dv)[p] = d_fv;
		} else {
			CUDA_CHECK_RETURN(cudaMemcpyAsync((*dv)[p]->input, hv[i]->input, n_input*sizeof(float), cudaMemcpyHostToDevice, stream));
			CUDA_CHECK_RETURN(cudaMemcpyAsync((*dv)[p]->target, hv[i]->target, n_output*sizeof(float), cudaMemcpyHostToDevice, stream));
		}
	}
	finish = clock();
	std::cout << "Copy time: " << ((float)finish-start)/CLOCKS_PER_SEC << std::endl;
}
