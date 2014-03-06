/*
 * GPUNet.cpp
 *
 *  Created on: Jan 5, 2014
 *      Author: trevor
 */

#include "GPUNet.h"
#include "NetTrainer.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <thrust/device_vector.h>
#include "curand_kernel.h"

/*
 * ------------ CUDA ------------
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

__device__ inline float sigmoid(float x) {
	return 1.0 / (1.0 + exp(-x));
}

__device__ float calc_output_gradient(float output, float target) {
	return output * (1 - output) * (target - output);
}

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
 * curand_setup
 * node init
 * weight init
 * delta init
 *
 */

__global__ void curand_setup(curandState *state) {
	unsigned int seed = (unsigned int)clock64();
	int id = threadIdx.x;
	curand_init(seed, id, 0, &state[id]);
}


/*
 * initialize nodes to 0 or 1 if bias
 * block(1), threads(n_nodes+1)
 */
__global__ void init_nodes_layer(float *nodes) {
	int i = threadIdx.x;
	if (i == blockDim.x-1)
		nodes[i] = 1;
	else
		nodes[i] = 0;
}

//block(1), threads(n_output)
//set all output nodes to 0
__global__ void init_nodes_output(float *output) {
	int i = threadIdx.x;
	output[i] = 0;
}

//block(1), threads(n_layer1+1, n_layer2)
__global__ void init_weights(float *weights, curandState *state) {
	// r is the range for random values
	__shared__ float r; r = 1.0 / sqrt((float)blockDim.x-1);
	__syncthreads();

	int node_l1 = threadIdx.x;
	int node_l2 = threadIdx.y;
	weights[blockDim.y*node_l1 + node_l2] = get_random_range(-r, r, blockDim.y*node_l1 + node_l2, state);
}

// block(1), threads(n_layer1+1, n_layer2)
__global__ void init_deltas(float *deltas) {
	int node_l1 = threadIdx.x;
	int node_l2 = threadIdx.y;

	deltas[blockDim.y*node_l1 + node_l2] = 0;
}



/*
 * --------------- Referencing and simple set function ---------------
 * set bias
 * set input/target[ref]
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

__device__ bool d_correct_result = true;
// called with blocks = (1), threads = (n_output)
// d_correct_result must be set to true before each call
// d_correct_result is copied back to host afterwards
__global__ void output_correct(float *output, float *target) {
	int i = threadIdx.x;
	if (d_correct_result && clamp(output[i]) != target[i]) {
		d_correct_result = false;
	}
}

__device__ float d_mse_sum = 0;
// called with blocks = (1), threads = (n_output)
// d_mse_sum must be set to 0 before each call
// d_mse_sum is copied back to host afterwards
__global__ void mse_sum(float *output, float *target) {
	int i = threadIdx.x;
	d_mse_sum += pow((output[i] - target[i]), 2);
}

/*
 * use reduction instead of writing repeatedly to global memory
 */
__global__ void mse_sum_v2(float *output, float *target, float *sums) {
	int i = threadIdx.x;
	sums[i] = pow((output[i] - target[i]), 2);

	__syncthreads();

	//TODO: do reduction in shared memory here if number of threads is less than 128 or so?
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
 * evoked with blocks(num nodes layer 2), threads(num nodes layer 1)
 *
 * sums holds the values of each term in the linear combination
 * n1t1, n1t2, ... n1tm, n2t1, n2t2, ... , n2tm, ...
 *
 *
 * bandwidth calculation:
 *
 * per thread,
 * 	bytes read: 4*2
 * 	bytes written: 4*1
 *
 * (n_input+1)*n_hidden threads
 *
 * Total: (n_input+1)*n_hidden*4*3
 */
__global__ void feed_forward_layer_v2(int n_layer1, int n_layer2, float* layer1, float* layer2, float* weights, float* sums) {
	int j = blockIdx.x; //the node in the next layer to compute
	int i = threadIdx.x + threadIdx.y * blockDim.x; //0;//the term in the linear combination to compute

	if (i > n_layer1) {
		return;
	}

	sums[(n_layer1+1)*j+i] = layer1[i] * weights[n_layer2*i + j];
}

__global__ void feed_forward_layer_v2_2(int n_layer1, int n_layer2, float* layer1, float* layer2, float* weights, float* sums) {

	unsigned int i = blockIdx.x * blockDim.x+threadIdx.x; // input node

	if (i <= n_layer1) {
		int j = i % n_layer2;
		sums[(n_layer1+1)*j+i] = layer1[i] * weights[n_layer2*i + j];
	}

}


__global__ void compute_activation(float* nodes, float *sums, int stride) {
	int i = threadIdx.x;
	nodes[i] = sigmoid(sums[i*stride]);
}

/*
 * generic version
 */
__global__ void compute_activation_v2(float* nodes, float *sums, int n_layer, int stride) {
	unsigned int i = blockIdx.x * blockDim.x+threadIdx.x; // input node

	if (i < n_layer)
		nodes[i] = sigmoid(sums[i*stride]);
}


/*
 * n_nodes is the number of nodes in the previous layer
 *
 * bandwidth calc:
 * 	bytes read: 4
 * 	bytes written: 4
 *
 */
__global__ void reduction(int n_nodes_prev, int n_nodes_next, int itr, float* sums) {
	//initialize dynamic shared memory for floor(n_nodes_prev/2.0)*n_nodes_next floats

	//get index
	int i = threadIdx.x;
	int index = i + blockIdx.x * blockDim.x; // * n_nodes_prev
	//if in correct term and not leftover
	if ((i % (int)powf(2,itr)) == 0 && ((i+1) % n_nodes_prev) != 0) {
		//printf("sums[%d] = %f, sums[%d+%d] = %f\n", i, sums[i], i, (int)powf(2, itr-1), sums[i+ (int)powf(2,itr-1)]);
		sums[index] += sums[index + (int)powf(2,itr-1)];
	}
}

/*
 * Shared memory reduction, handles threads more efficiently.
 */
__global__ void split_reduce(int n, int offset, float *g_idata, float *g_odata) {
	extern __shared__ float sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x + offset;

	if (i < n + offset) { // if in our range of data
		//printf("i = %d\n", i);
		sdata[tid] = g_idata[i];
		__syncthreads();

		for (unsigned int s = 1; s < blockDim.x; s *= 2) {
			int index = 2 * s * tid;
			if (index < blockDim.x && (index + s) < n) {
				//printf("sdata[%d] = %f,  sdata[%d] = %f\n", index, sdata[index], index+s, sdata[index+s]);
				sdata[index] += sdata[index + s];
			}
			__syncthreads();
		}
		if (tid == 0) g_odata[blockIdx.x+offset] = sdata[0];
	}
}


/*
 *
 *
 * ------------ backprop kernels ---------
 *
 *
 */

/*
 * called with threads(n_output)
 */
__global__ void output_error_gradients(float* output, float* target, float* output_err_gradients) {
	int i = threadIdx.x;
	output_err_gradients[i] = calc_output_gradient(output[i], target[i]);
	//printf("out_err_grad[%d] = %f, output = %f, target = %f\n", i, output_err_gradients[i], output[i], target[i]);
}

/*
 * called generically, pow of 2 threads
 */
__global__ void output_error_gradients_v2(float* output, float* target, float* output_err_gradients, int no) {
	unsigned int i = blockIdx.x * blockDim.x+threadIdx.x;

	if (i < no)
		output_err_gradients[i] = calc_output_gradient(output[i], target[i]);
	//printf("out_err_grad[%d] = %f, output = %f, target = %f\n", i, output_err_gradients[i], output[i], target[i]);
}

/*
 * called with threads = (nh+1, no, 1)
 */
__global__ void update_hidden_output_deltas(int no, float l_rate, float momentum,
		float* hidden, float* output_err_gradients, float* delta_ho) {

	int j = threadIdx.x; //hidden node
	int k = threadIdx.y; //output node

	//This probably doesn't improve much
	//I assume that what really happens is every node writes the same value into shared memory
	//so really every thread is still doing the work
	//__shared__ float out_err_grad_k; out_err_grad_k = output_err_gradients[k];
	//__syncthreads();

	delta_ho[no*j + k] = l_rate * hidden[j] * output_err_gradients[k] + momentum * delta_ho[no*j + k];
	//printf("delta_ho(%d, %d) = %f, l_rate = %f, hidden[%d] = %f, out_err_gradients[%d] = %f, momentum = %f\n",
	//		j, k, delta_ho[no*j+k], l_rate, j, hidden[j], k, output_err_gradients[k], momentum);

}


/*
 * called generically with power of 2 threads
 */
__global__ void update_hidden_output_deltas_v2(int nh, int no, float l_rate, float momentum,
		float* hidden, float* output_err_gradients, float* delta_ho) {

	unsigned int x = blockIdx.x * blockDim.x+threadIdx.x;

	int j = x % nh; //input node
	int k = x % no; //hidden node

	if (no*j+k < (nh+1)*no)
		delta_ho[no*j + k] = l_rate * hidden[j] * output_err_gradients[k] + momentum * delta_ho[no*j + k];
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
 * called with threads = (nh)
 */
__global__ void hidden_error_gradients(int no, float* hidden, float* d_ho_weights, float* hidden_err_gradients, float* output_err_gradients) {
	int j = threadIdx.x;
	hidden_err_gradients[j] = calc_hidden_gradient(j, no, hidden, d_ho_weights, output_err_gradients);
	//printf("hidden_err_grad[%d] = %f\n", j, hidden_err_gradients[j]);
}

/*
 * called generically, pow of 2 threads
 */
__global__ void hidden_error_gradients_v2(int nh, int no, float* hidden, float* d_ho_weights, float* hidden_err_gradients, float* output_err_gradients) {
	unsigned int j = blockIdx.x * blockDim.x+threadIdx.x;

	if (j < (nh+1)*no)
		hidden_err_gradients[j] = calc_hidden_gradient(j, no, hidden, d_ho_weights, output_err_gradients);
	//printf("hidden_err_grad[%d] = %f\n", j, hidden_err_gradients[j]);
}


/*
 * called with blocks(no), threads(nh)
 *
 * reduce this list and then call calc gradients
 *
 * TODO: generalize this to pow 2 threads / blocks
 */
__global__ void hidden_error_gradients_v3(int no, float *sums, float *d_ho_weights, float *output_err_gradients) {
	int j = threadIdx.x;
	int k = blockIdx.x;

	sums[j*no + k] = d_ho_weights[j*no + k] * output_err_gradients[k];
}

/*
 * called with threads(nh)
 *
 * TODO: generalize this to pow 2 threads / blocks
 */
__global__ void calc_gradients(float *sums, float *hidden, float*hidden_err_gradients) {
	int i = threadIdx.x;
	hidden_err_gradients[i] = hidden[i] * (1 - hidden[i]) * sums[i*blockDim.x];
}

__global__ void update_input_hidden_deltas(int nh, float l_rate, float momentum,
		float* input, float* hidden_err_gradients, float* delta_ih) {

	int i = threadIdx.y; //input node
	int j = threadIdx.x; //hidden node

	delta_ih[nh*i + j] = l_rate * input[i] * hidden_err_gradients[j] + momentum * delta_ih[nh*i + j];

	//printf("delta_ho(%d, %d) = %f, l_rate = %f, input[%d] = %f, hidden_err_gradients[%d] = %f, momentum = %f\n",
	//			i, j, delta_ih[nh*i + j], l_rate, i, input[i], j, hidden_err_gradients[j], momentum);
}

/*
 * called with any number of blocks / threads
 * normally, 128 or other power of 2
 */
//TODO: perhaps there is a way to store the hidden_err_gradient[j] in shared memory
__global__ void update_input_hidden_deltas_v2(int ni, int nh, float l_rate, float momentum,
		float* input, float* hidden_err_gradients, float* delta_ih) {
	unsigned int x = blockIdx.x * blockDim.x+threadIdx.x;

	int i = x % ni; //input node
	int j = x % nh; //hidden node

	if (nh*i+j < (ni+1)*nh)
		delta_ih[nh*i + j] = l_rate * input[i] * hidden_err_gradients[j] + momentum * delta_ih[nh*i + j];

	//printf("delta_ho(%d, %d) = %f, l_rate = %f, input[%d] = %f, hidden_err_gradients[%d] = %f, momentum = %f\n",
	//			i, j, delta_ih[nh*i + j], l_rate, i, input[i], j, hidden_err_gradients[j], momentum);
}



/*
 * weight update
 */

//blocks(n_layer1), threads(n_layer0+1)
__global__ void update_weights(float *d_weights, float* deltas) {
	int j = blockIdx.x; //layer1
	int i = threadIdx.x; //layer0

	d_weights[i*gridDim.x + j] += deltas[i*gridDim.x + j];
}

/*
 * called generically with power of 2 threads
 */
__global__ void update_weights_v2(int n1, int n2, float *d_weights, float *deltas) {
	unsigned int x = blockIdx.x * blockDim.x+threadIdx.x;

	int i = x % n1; //layer 1 node
	int j = x % n2; //layer 2 node

	if (i*n2+j < (n1+1)*n2)
		d_weights[i*n2 + j] += deltas[i*n2 + j];
}


//blocks(n_output), threads(n_hidden+1)
__global__ void update_weights_ho(int no, float* d_ho_weights, float* deltas_ho) {
	int k = blockIdx.x; //output
	int j = threadIdx.x; //hidden

	d_ho_weights[j*no + k] += deltas_ho[j*no + k];
}

//blocks(n_hidden), threads(n_input+1)
__global__ void update_weights_ih(int nh, float* d_ih_weights, float* deltas_ih) {
	int k = blockIdx.x; //hidden
	int j = threadIdx.x; //input

	d_ih_weights[j*nh + k] += deltas_ih[j*nh + k];
}

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

/*
 *
 * --------- Debugging ------------
 *
 */

__global__ void print_floats(int n_input, float* d_input_2, FeatureVector *d_fv) {
	printf("d_fv.input: %f\n", d_fv->input[0]);
	printf("d_fv.input: %f\n", d_fv->input[1]);
	printf("d_fv.input: %f\n", d_fv->input[2]);
	for(int i = 0; i < n_input; ++i) {
		printf("%d: %f\n", i, d_input_2[i]);
	}
}

__global__ void print_floats2(int n_input, FeatureVector *d_fv) {
	printf("d_fv.input: %f\n", d_fv->input[0]);
	printf("d_fv.input: %f\n", d_fv->input[1]);
}

__global__ void print_all(int n_input, int n_output, FeatureVector **dv) {
	for (int i = 0; i < 4; ++i) {
		printf("Pattern %d\n", i);
		printf("Input: ");
		for (int j = 0; j < n_input; ++j) {
			printf("%f ", dv[i]->input[j]);
		}
		printf("\nTarget: ");
		for (int k = 0; k < n_output; ++k) {
			printf("%f ", dv[i]->target[k]);
		}
		printf("\n");
	}

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
 * ---------- public -------------
 */

GPUNet::GPUNet(int ni, int no, GPUNet::NetworkStructure net_type) {
	n_input = ni;
	n_output = no;

	if (net_type == GPUNet::STANDARD) {
		n_hidden = ceil(2.0/3.0*ni);
	} else if (net_type == GPU_ARCH_OPT) {
		n_hidden = 128 / (2.0/3.0*ni) * 128;
	} else {
		std::cerr << "Invalid network type: " << net_type << std::endl;
		exit(1);
	}

	max_epochs = GPU_MAX_EPOCHS;
	l_rate = GPU_LEARNING_RATE;
	momentum = GPU_MOMENTUM;
	desired_acc = GPU_DESIRED_ACCURACY;
	n_gpus = N_GPUS;

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

	d_input_ptr = NULL;
	d_target_ptr = NULL;

	d_ih_weights = NULL;
	d_ho_weights = NULL;

	d_ih_deltas = NULL;
	d_ho_deltas = NULL;

	d_hid_err_gradients = NULL;
	d_out_err_gradients = NULL;

	/*
	 * host validation
	 */
	h_output = new float[n_output];
	h_ih_weights = new float[(n_input+1)*n_hidden];
	h_ho_weights = new float[(n_hidden+1)*n_output];

	//init gpu mem to 0 for each gpu
	gpu_mem = new size_t[n_gpus];
	for (int i = 0; i < n_gpus; ++i) {
		gpu_mem[i] = 0;
	}
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

	delete[] h_output;
	delete[] gpu_mem;
}

/*
 * -------------- public ---------------
 */

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

	std::cout << "Memory allocated on device." << std::endl;
}

/*
 * Note: assumes sizes of networks are the same
 * This is for testing purposes so that
 * I can have identical networks.
 */
void GPUNet::init_from_net(Net &net, NetData &d) {
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

	//init deltas to 0
	dim3 ih_threads(n_input+1, n_hidden);
	dim3 ho_threads(n_hidden+1, n_output);
	init_deltas<<<1, ih_threads>>>(d_ih_deltas);
	init_deltas<<<1, ho_threads>>>(d_ho_deltas);

	std::cout << "data copied to device\n\n";
}


void GPUNet::init_net() {
	//init nodes to all 0
	init_nodes_layer<<<1, n_input+1>>>(d_input);
	init_nodes_layer<<<1, n_hidden+1>>>(d_hidden);
	init_nodes_output<<<1, n_output>>>(d_output);

	//init weights to random vals
	curandState *state;
	CUDA_CHECK_RETURN(cudaMalloc(&state, (n_input+1)*n_hidden*sizeof(curandState)));
	curand_setup<<<1, (n_input+1)*n_hidden>>>(state);
	dim3 ih_threads(n_input+1, n_hidden);
	init_weights<<<1, ih_threads>>>(d_ih_weights, state);

	CUDA_CHECK_RETURN(cudaMalloc(&state, (n_hidden+1)*n_output*sizeof(curandState)));
	curand_setup<<<1, (n_hidden+1)*n_output>>>(state);
	dim3 ho_threads(n_hidden+1, n_output);
	init_weights<<<1, ho_threads>>>(d_ho_weights, state);

	//init deltas to 0
	init_deltas<<<1, ih_threads>>>(d_ih_deltas);
	init_deltas<<<1, ho_threads>>>(d_ho_deltas);

	std::cout << "net initialized" << std::endl;
}

void GPUNet::set_learning_rate(float lr) {
	l_rate = lr;
}

void GPUNet::set_momentum(float m) {
	momentum = m;
}

void GPUNet::set_training_params(float lr, float m) {
	l_rate = lr;
	momentum = m;
}

void GPUNet::set_max_epochs(int me) {
	max_epochs = me;
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
void GPUNet::write_net(std::string fname) {
	std::ofstream of(fname.c_str());

	float *ih_weights = new float[(n_input+1)*(n_hidden)];
	float *ho_weights = new float[(n_hidden+1)*(n_output)];

	CUDA_CHECK_RETURN(cudaMemcpy(ih_weights, d_ih_weights, (n_input+1)*(n_hidden)*sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(ho_weights, d_ho_weights, (n_hidden+1)*(n_output)*sizeof(float), cudaMemcpyDeviceToHost));

	if (of.is_open()) {
		of << "num_epochs=" << epoch << "\n";
		of << "max_epochs=" << max_epochs << "\n";
		of << "num_layers=" << 3 << "\n";
		of << "n_layer_0=" << n_input << "\n";
		of << "n_layer_1=" << n_hidden << "\n";
		of << "n_layer_2=" << n_output << "\n";
		of << "l_rate=" << l_rate << "\n";
		of << "momentum=" << momentum << "\n";
		of << "desired_acc=" << desired_acc << "\n";
		of << "tset_acc=" << trainingSetAccuracy << "\n";
		of << "gset_acc=" << generalizationSetAccuracy << "\n";
		of << "vset_acc=" << validationSetAccuracy << "\n";
		of << "tset_mse=" << trainingSetMSE << "\n";
		of << "gset_mse=" << generalizationSetMSE << "\n";
		of << "vset_mse=" << validationSetMSE << "\n";
		of << "weights_ih=(";
		for (int i = 0, l = (n_input+1)*n_hidden; i < l; ++i) {
			of << ih_weights[i];
			if (i != l-1)
				of << ",";
		}
		of << ")\n";
		of << "weights_ho=(";
		for (int i = 0, l = (n_hidden+1)*n_output; i < l; ++i) {
			of << ho_weights[i];
			if (i != l-1)
				of << ",";
		}
		of << ")";

		of.flush();
		of.close();
	} else {
		std::cout << "Could not write file!" << std::endl;
	}
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



void GPUNet::train_net(TrainingDataSet *tset) {
	std::cout << std::endl << " Neural Network Training Starting: " << std::endl
			<< "==========================================================================" << std::endl
			<< " LR: " << l_rate << ", Momentum: " << momentum << ", Max Epochs: " << max_epochs << std::endl
			<< " " << n_input << " Input Neurons, " << n_hidden << " Hidden Neurons, " << n_output << " Output Neurons" << std::endl
			<< "==========================================================================" << std::endl << std::endl;

	FeatureVector** d_training_set;
	//FeatureVector* d_generalization_set;
	//FeatureVector* d_validation_set;
	//TODO: ultimately replace this with a network partitioning call
	if (dataset_size(tset) + current_mem_usage(get_current_device()) < total_dev_mem(get_current_device())) {

		start = clock();
		//copy_to_device(tset->training_set, &d_training_set);
		//copy_to_device_biased(tset->training_set, &d_training_set);
		//copy_to_device_host_array(tset->training_set, &d_training_set);
		//copy_to_device_host_array_biased(tset->training_set, &d_training_set);
		copy_to_device_host_array_ptrs_biased(tset->training_set, &d_training_set);
		finish = clock();

		std::cout << "Copying entire dataset to device: " << ((float)finish-start)/CLOCKS_PER_SEC << std::endl;


		//for (int i = 0; i < tset->training_set.size(); ++i) {
		//	print_input<<<1,1>>>(n_input, d_training_set[i]->input);
		//	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		//	print_target<<<1,1>>>(n_output, d_training_set[i]->target);
		//	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		//}
	}

	//return;

	//reset epoch and log counters
	epoch = 0;

	//train network using training dataset for training and generalization dataset for testing
	while ((trainingSetAccuracy < desired_acc) && epoch < max_epochs) {
	//while ((trainingSetAccuracy < desired_acc || generalizationSetAccuracy < desired_acc) && epoch < max_epochs) {
		//store previous accuracy
		float previousTAccuracy = trainingSetAccuracy;
		float previousGAccuracy = generalizationSetAccuracy;

		//use training set to train network
		//run_training_epoch(tset->training_set);
		//std::cout << "Calling run_training_epoch_dev" << std::endl;
		run_training_epoch_dev(d_training_set, tset->training_set.size());

		//return;

		//get generalization set accuracy and MSE
		//get_set_accuracy_mse(tset->generalization_set, &generalizationSetAccuracy, &generalizationSetMSE);
		//get_set_accuracy_mse_dev(d_generalization_set, tset->generalization_set.size(), &generalizationSetAccuracy, &generalizationSetMSE);

		//print out change in training /generalization accuracy (only if a change is greater than a percent)
		//if (ceil(previousTAccuracy) != ceil(trainingSetAccuracy) || ceil(previousGAccuracy) != ceil(generalizationSetAccuracy)) {
			std::cout << "Epoch: " << epoch;
			std::cout << "; Test Set Acc:" << trainingSetAccuracy << "%, MSE: " << trainingSetMSE;
			std::cout << ";\tGSet Acc:" << generalizationSetAccuracy << "%, MSE: " << generalizationSetMSE << std::endl;
		//}

		//once training set is complete increment epoch
		++epoch;
	}

	//get validation set accuracy and MSE
	//get_set_accuracy_mse(tset->validation_set, &validationSetAccuracy, &validationSetMSE);

	//out validation accuracy and MSE
	std::cout << std::endl << "Training Complete. Elapsed Epochs: " << epoch << std::endl;
	std::cout << "\tValidation Set Accuracy: " << validationSetAccuracy << std::endl;
	std::cout << "\tValidation Set MSE: " << validationSetMSE << std::endl << std::endl;
}

void GPUNet::get_set_accuracy_mse(thrust::host_vector<FeatureVector*> set, float* s_acc, float* s_mse) {
	int incorrect_patterns = 0;
	float mse = 0, mse_tmp = 0;
	bool correct_result = true;

	//TODO: copy multiple patters at once so bandwidth is not a limiting factor
	for (unsigned int i = 0; i < set.size(); ++i) {
		mse_tmp = 0;
		correct_result = true;

		//copy pattern to input
		CUDA_CHECK_RETURN(cudaMemcpy(d_input, set[i]->input, (n_input+1)*sizeof(float), cudaMemcpyHostToDevice));

		//copy target to dev
		CUDA_CHECK_RETURN(cudaMemcpy(d_target, set[i]->target, (n_output)*sizeof(float), cudaMemcpyHostToDevice));

		//feed forward input
		feed_forward_v1();

		CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_correct_result, &correct_result, sizeof(correct_result), 0, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_mse_sum, &mse_tmp, sizeof(mse_tmp), 0, cudaMemcpyHostToDevice));
		output_correct<<<1, n_output>>>(d_output, d_target);
		mse_sum<<<1, n_output>>>(d_output, d_target);
		CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&correct_result, d_correct_result, sizeof(correct_result), 0, cudaMemcpyDeviceToHost));
		CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&mse_tmp, d_mse_sum, sizeof(mse_tmp), 0, cudaMemcpyDeviceToHost));
		if (!correct_result)
			++incorrect_patterns;
		mse += mse_tmp;
	}

	*s_acc = 100 - ((float)incorrect_patterns/set.size() * 100);
	*s_mse = mse / (n_output * set.size());
}

void GPUNet::run_training_epoch(thrust::host_vector<FeatureVector*> feature_vecs) {
	int incorrect_patterns = 0;
	float mse = 0, mse_tmp = 0;
	bool correct_result = true;

	for (unsigned int i = 0; i < feature_vecs.size(); ++i) {
		mse_tmp = 0;
		correct_result = true;

		//copy pattern to input
		CUDA_CHECK_RETURN(cudaMemcpy(d_input, feature_vecs[i]->input, (n_input+1)*sizeof(float), cudaMemcpyHostToDevice));

		//copy target to dev
		CUDA_CHECK_RETURN(cudaMemcpy(d_target, feature_vecs[i]->target, (n_output)*sizeof(float), cudaMemcpyHostToDevice));

		//feed forward input
		feed_forward_v1();

		CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_correct_result, &correct_result, sizeof(correct_result), 0, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_mse_sum, &mse_tmp, sizeof(mse_tmp), 0, cudaMemcpyHostToDevice));
		output_correct<<<1, n_output>>>(d_output, d_target);
		mse_sum<<<1, n_output>>>(d_output, d_target);
		CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&correct_result, d_correct_result, sizeof(correct_result), 0, cudaMemcpyDeviceToHost));
		CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&mse_tmp, d_mse_sum, sizeof(mse_tmp), 0, cudaMemcpyDeviceToHost));
		if (!correct_result)
			++incorrect_patterns;
		mse += mse_tmp;

		//std::cout << "Correct result: " << correct_result << ", mse_tmp: " << mse_tmp << std::endl;
		//backprop target
		backprop_v1();
	}


	//std::cout << "MSE sum: " << mse << std::endl;
	//std::cout << "inc patterns: " << incorrect_patterns << std::endl;
	//update training accuracy and MSE
	trainingSetAccuracy = 100 - ((float)incorrect_patterns/feature_vecs.size() * 100);
	trainingSetMSE = mse / (n_output * feature_vecs.size());
}

void GPUNet::get_set_accuracy_mse_dev(FeatureVector **feature_vecs, size_t n_features, float* s_acc, float* s_mse) {
	int incorrect_patterns = 0;
	float mse = 0, mse_tmp = 0;
	bool correct_result = true;

	//TODO: copy multiple patters at once so bandwidth is not a limiting factor
	for (unsigned int i = 0; i < n_features; ++i) {
		mse_tmp = 0;
		correct_result = true;

		//feed forward input
		feed_forward_v1_2(feature_vecs[i]->input);

		CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_correct_result, &correct_result, sizeof(correct_result), 0, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_mse_sum, &mse_tmp, sizeof(mse_tmp), 0, cudaMemcpyHostToDevice));
		output_correct<<<1, n_output>>>(d_output, feature_vecs[i]->target);
		mse_sum<<<1, n_output>>>(d_output, feature_vecs[i]->target);
		CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&correct_result, d_correct_result, sizeof(correct_result), 0, cudaMemcpyDeviceToHost));
		CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&mse_tmp, d_mse_sum, sizeof(mse_tmp), 0, cudaMemcpyDeviceToHost));
		if (!correct_result)
			++incorrect_patterns;
		mse += mse_tmp;
	}

	*s_acc = 100 - ((float)incorrect_patterns/n_features * 100);
	*s_mse = mse / (n_output * n_features);
}

void GPUNet::run_training_epoch_dev(FeatureVector **feature_vecs, size_t n_features) {
	int incorrect_patterns = 0;
	float mse = 0, mse_tmp = 0;
	bool correct_result = true;

	for (size_t i = 0; i < n_features; ++i) {
		mse_tmp = 0;
		correct_result = true;

		//print_input<<<1,1>>>(n_input, feature_vecs[i]->input);
		//CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		//print_target<<<1,1>>>(n_output, feature_vecs[i]->target);
		//CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		feed_forward_v1_2(feature_vecs[i]->input);

		//TODO: maintain these variables on the GPU side
		CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_correct_result, &correct_result, sizeof(correct_result), 0, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_mse_sum, &mse_tmp, sizeof(mse_tmp), 0, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		output_correct<<<1, n_output>>>(d_output, feature_vecs[i]->target);
		mse_sum<<<1, n_output>>>(d_output, feature_vecs[i]->target);
		CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&correct_result, d_correct_result, sizeof(correct_result), 0, cudaMemcpyDeviceToHost));
		CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&mse_tmp, d_mse_sum, sizeof(mse_tmp), 0, cudaMemcpyDeviceToHost));
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		if (!correct_result) ++incorrect_patterns;
		mse += mse_tmp;

		//std::cout << "Correct result: " << correct_result << ", mse_tmp: " << mse_tmp << std::endl;

		backprop_v2(feature_vecs[i]->input, feature_vecs[i]->target);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	}

	//std::cout << "MSE sum: " << mse << std::endl;
	//std::cout << "inc patterns: " << incorrect_patterns << std::endl;
	//update training accuracy and MSE
	trainingSetAccuracy = 100 - ((float)incorrect_patterns/n_features * 100);
	trainingSetMSE = mse / (n_output * n_features);
}

/*
 * Reduce sums from len to n sums. Assumes len is a multiple of n.
 */
float* GPUNet::reduce(int n, int len, float* d_sums, float *d_y) {
	int step = len / n;

	float *res;
	for (int i = n-1; i >= 0; --i) {
		res = execute_split_reduction(step, i*step, d_sums, d_y);
	}
	return res;
}

/*
 * n is number of elements to sum
 * offset is where to start in the list
 * d_x is original list
 */
float* GPUNet::execute_split_reduction(int n, int offset, float *d_x, float *d_y) {
	bool result_in_y = false;
	int threads = 128;
	int blocks = (n+threads-1);

	if (n >= threads) {
		do {
			blocks /= threads;
			if (result_in_y)
				split_reduce<<<blocks, threads, threads*sizeof(float)>>>(n, offset, d_y, d_x);
			else
				split_reduce<<<blocks, threads, threads*sizeof(float)>>>(n, offset, d_x, d_y);
			result_in_y = !result_in_y;
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		} while (blocks/threads >= threads);
		if (result_in_y)
			//reduce0<<<1, threads, threads*sizeof(float)>>>(blocks, d_y, d_x);
			split_reduce<<<1, blocks, blocks*sizeof(float)>>>(n, offset, d_y, d_x);
		else
			//reduce0<<<1, threads, threads*sizeof(float)>>>(blocks, d_x, d_y);
			split_reduce<<<1, blocks, blocks*sizeof(float)>>>(n, offset, d_x, d_y);
		result_in_y = !result_in_y;
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	} else {
		split_reduce<<<1, n, n*sizeof(float)>>>(n, offset, d_x, d_y);
		result_in_y = !result_in_y;
	}

	if (result_in_y) {
		return d_y;
	} else {
		return d_x;
	}
}



void GPUNet::backprop_v1() {
	output_error_gradients<<<1, n_output>>>(d_output, d_target, d_out_err_gradients);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	/*
	 * called with threads = (nh+1, no, 1)
	 */
	dim3 hid_out_deltas(n_hidden+1, n_output);
	update_hidden_output_deltas<<<1, hid_out_deltas>>>(n_output, l_rate, momentum, d_hidden, d_out_err_gradients, d_ho_deltas);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	/*
	 * called with threads = (nh)
	 */
	hidden_error_gradients<<<1, n_hidden>>>(n_output, d_hidden, d_ho_weights,
			d_hid_err_gradients, d_out_err_gradients);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	update_weights_ho<<<n_output, n_hidden+1>>>(n_output, d_ho_weights, d_ho_deltas);

	dim3 in_hid_deltas(n_input+1, n_hidden);
	update_input_hidden_deltas<<<1, in_hid_deltas>>>(n_hidden, l_rate, momentum,
			d_input, d_hid_err_gradients, d_ih_deltas);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	update_weights_ih<<<n_hidden, n_input+1>>>(n_hidden, d_ih_weights, d_ih_deltas);
}



void GPUNet::backprop_v2(float *d_inp, float *d_tar) {
	int n_threads = 128;

	output_error_gradients_v2<<<(n_output+n_threads-1)/n_threads, n_threads>>>(d_output, d_tar, d_out_err_gradients, n_output);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	update_hidden_output_deltas_v2<<<((n_output*(n_hidden+1))+n_threads-1)/n_threads, n_threads>>>(n_hidden, n_output, l_rate, momentum, d_hidden, d_out_err_gradients, d_ho_deltas);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	hidden_error_gradients_v2<<<(n_hidden+n_threads-1)/n_threads, n_threads>>>(n_hidden, n_output, d_hidden, d_ho_weights,
			d_hid_err_gradients, d_out_err_gradients);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	update_weights_v2<<<((n_output*(n_hidden+1))+n_threads-1)/n_threads, n_threads>>>(n_hidden, n_output, d_ho_weights, d_ho_deltas);

	update_input_hidden_deltas_v2<<<((n_hidden*(n_input+1))+n_threads-1)/n_threads, n_threads>>>(n_input, n_hidden, l_rate, momentum,
				d_inp, d_hid_err_gradients, d_ih_deltas);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	update_weights_v2<<<((n_hidden*(n_input+1))+n_threads-1)/n_threads, n_threads>>>(n_input, n_hidden, d_ih_weights, d_ih_deltas);
}

void GPUNet::feed_forward_v1() {
	//std::cout << "feed forward GPU method 1" << std::endl;
	//start = clock();
	feed_forward_layer_v1<<<1, n_hidden>>>(n_input, n_hidden, d_input, d_hidden, d_ih_weights);
	// must finish previous layer before computing next
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	feed_forward_layer_v1<<<1, n_output>>>(n_hidden, n_output, d_hidden, d_output, d_ho_weights);

	//sync before measuring time
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	//finish = clock();
	//std::cout << "kernel method 1 evaluated: " << ((float)(finish-start)) / CLOCKS_PER_SEC << "s\n\n";
}

void GPUNet::feed_forward_v1_2(float *d_inp) {
	int threads = 128;
	feed_forward_layer_v1_2<<<(n_hidden+threads-1)/threads, threads>>>(n_input, n_hidden, d_inp, d_hidden, d_ih_weights);
	// must finish previous layer before computing next
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	feed_forward_layer_v1_2<<<(n_output+threads-1)/threads, threads>>>(n_hidden, n_output, d_hidden, d_output, d_ho_weights);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}


void GPUNet::feed_forward_v2() {
	dim3 gridm2l1(n_hidden);
	dim3 threadsm2l1 = get_threadsm2l1();
	//std::cout << "threads layer 1: (" << threadsm2l1.x << " " << threadsm2l1.y << " " << threadsm2l1.z << ")" << std::endl;

	//float *a = new float[(n_input+1)*n_hidden];
	float* d_sums_l1, *d_y;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_sums_l1, (n_input+1)*n_hidden*sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_y, (n_input+1)*n_hidden*sizeof(float)));

	feed_forward_layer_v2<<<gridm2l1, threadsm2l1>>>(n_input, n_hidden, d_input, d_hidden, d_ih_weights, d_sums_l1);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	d_sums_l1 = reduce(n_hidden, (n_input+1)*n_hidden, d_sums_l1, d_y);

	compute_activation<<<1, n_hidden>>>(d_hidden, d_sums_l1, n_input+1);

	// must finish previous layer before computing next
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaFree(d_sums_l1));
	//GPUNet::add_gpu_mem(-(n_input+1)*n_hidden*sizeof(float));

	//grid size must be >= # nodes in next layer
	dim3 gridm2l2(n_output);
	//1 thread per grid
	dim3 threadsm2l2 = get_threadsm2l2();
	//std::cout << "threads layer 2: (" << threadsm2l2.x << " " << threadsm2l2.y << " " << threadsm2l2.z << ")" << std::endl;

	float *d_sums_l2;
	CUDA_CHECK_RETURN(cudaMalloc(&d_sums_l2, n_hidden*n_output*sizeof(float)));
	CUDA_CHECK_RETURN(cudaMemset(d_y, 0, (n_input+1)*n_hidden*sizeof(float)));
	//GPUNet::add_gpu_mem((n_hidden+1)*n_output*sizeof(float));

	feed_forward_layer_v2<<<gridm2l2, threadsm2l2>>>(n_hidden, n_output, d_hidden, d_output, d_ho_weights, d_sums_l2);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	d_sums_l2 = reduce(n_output, (n_hidden+1)*n_output, d_sums_l2, d_y);

	compute_activation<<<1, n_output>>>(d_output, d_sums_l2, n_hidden+1);

	//sync before measuring time
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaFree(d_sums_l2));
	CUDA_CHECK_RETURN(cudaFree(d_y));
	//GPUNet::add_gpu_mem(-(n_hidden+1)*n_output*sizeof(float));
}

void GPUNet::feed_forward_v2_2() {
	int threads = 128;

	float* d_sums_l1, *d_y;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_sums_l1, (n_input+1)*n_hidden*sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_y, (n_input+1)*n_hidden*sizeof(float)));

	feed_forward_layer_v2_2<<<(n_input+threads-1)/threads, threads>>>(n_input, n_hidden, d_input, d_hidden, d_ih_weights, d_sums_l1);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	d_sums_l1 = reduce(n_hidden, (n_input+1)*n_hidden, d_sums_l1, d_y);

	compute_activation_v2<<<(n_hidden+threads-1)/n_hidden, threads>>>(d_hidden, d_sums_l1, n_hidden, n_input+1);

	// must finish previous layer before computing next
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaFree(d_sums_l1));
	//GPUNet::add_gpu_mem(-(n_input+1)*n_hidden*sizeof(float));

	float *d_sums_l2;
	CUDA_CHECK_RETURN(cudaMalloc(&d_sums_l2, n_hidden*n_output*sizeof(float)));
	CUDA_CHECK_RETURN(cudaMemset(d_y, 0, n_input*n_hidden*sizeof(float)));
	//GPUNet::add_gpu_mem((n_hidden+1)*n_output*sizeof(float));

	feed_forward_layer_v2_2<<<(n_hidden+threads-1)/threads, threads>>>(n_hidden, n_output, d_hidden, d_output, d_ho_weights, d_sums_l2);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	d_sums_l2 = reduce(n_output, (n_hidden+1)*n_output, d_sums_l2, d_y);
	compute_activation_v2<<<(n_output+threads-1)/n_output, threads>>>(d_output, d_sums_l2, n_output, n_hidden+1);

	//sync before measuring time
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaFree(d_sums_l2));
	CUDA_CHECK_RETURN(cudaFree(d_y));
	//GPUNet::add_gpu_mem(-(n_hidden+1)*n_output*sizeof(float));
}

bool GPUNet::validate_output(float* desired_output) {
	//copy output back to host
	CUDA_CHECK_RETURN(cudaMemcpy(h_output, d_output, n_output*sizeof(float), cudaMemcpyDeviceToHost));

	for (int i = 0; i < n_output; ++i) {
		std::cout << "actual = " << desired_output[i] << ", calc = " << h_output[i] << std::endl;
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

	for (int i = 0; i < n_input; ++i) {
		std::cout << "i = " << i << std::endl;
		std::cout << d.get_training_dataset()->training_set[0]->input[i] << std::endl;
	}
	std::cout << std::endl;

	std::cout << "feed forward CPU" << std::endl;
	start = clock();
	net.feed_forward(d.get_training_dataset()->training_set[0]->input);
	finish = clock();
	std::cout << "feed forward CPU time: " << ((float)(finish-start)) / CLOCKS_PER_SEC << "s\n\n";
	//net.print_network();

	std::cout << "Testing method 1" << std::endl;
	feed_forward_v1();
	std::cout << "Validates: " << validate_output(net.outputNeurons) << "\n";
	CUDA_CHECK_RETURN(cudaMemset(d_output, 0, n_output*sizeof(float)));

	print_net();

	std::cout << "Testing method 1.2" << std::endl;
	FeatureVector **dv;
	GPUNet::copy_to_device_host_array_ptrs_biased(d.get_training_dataset()->training_set, &dv);
	feed_forward_v1_2(dv[0]->input);
	std::cout << "Validates: " << validate_output(net.outputNeurons) << "\n";
	net.print_network();
	print_net();
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
	//clock_t start, finish;

	//copy target to dev
	CUDA_CHECK_RETURN(cudaMemcpy(d_target, d.get_training_dataset()->training_set[0]->target, (n_output)*sizeof(float), cudaMemcpyHostToDevice));

	NetTrainer nt(&net);
	//std::cout << "CPU net 0" << std::endl;
	//net.print_network();

	net.feed_forward(d.get_training_dataset()->training_set[0]->input);
	//std::cout << "CPU net 1" << std::endl;
	//net.print_network();

	nt.backprop(d.get_training_dataset()->training_set[0]->target);
	nt.update_weights();
	//std::cout << "CPU net 2" << std::endl;
	//net.print_network();

	std::cout << "Testing backprop_v2" << std::endl;
	//std::cout << std::endl << "GPU net 0" << std::endl;

	FeatureVector **dv;
	GPUNet::copy_to_device_host_array_ptrs_biased(d.get_training_dataset()->training_set, &dv);

	feed_forward_v1_2(dv[0]->input);
	//std::cout << "GPU net 1" << std::endl;
	//print_net();
	//std::cout << std::endl;

	//std::cout << "GPU net 2" << std::endl;
	backprop_v2(dv[0]->input, dv[0]->target);
	//print_net();
	//std::cout << std::endl;
	std::cout << "Validates: " << validate_weights(net.wInputHidden, net.wHiddenOutput) << std::endl;


	net.feed_forward(d.get_training_dataset()->training_set[1]->input);
	nt.backprop(d.get_training_dataset()->training_set[1]->target);
	nt.update_weights();
	feed_forward_v1_2(dv[1]->input);
	backprop_v2(dv[1]->input, dv[1]->target);


	std::cout << "Validates: " << validate_weights(net.wInputHidden, net.wHiddenOutput) << std::endl;
}


void GPUNet::test_reduction() {
	/*
	 * Testing with 4, easy since power of 2
	 */

	std::cout << std::endl << "Reduce array length 4" << std::endl;

	float a[] = {.25, .5, .75, 1};
	float *d_a;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_a, 4*sizeof(float)));
	CUDA_CHECK_RETURN(cudaMemcpy(d_a, &a, 4*sizeof(float), cudaMemcpyHostToDevice));
	reduction<<<1, 4>>>(4, 0, 1, d_a);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaMemcpy(&a, d_a, 4*sizeof(float), cudaMemcpyDeviceToHost));
	for (int i = 0; i < 4; ++i) {
		std::cout << a[i] << " ";
	}
	std::cout << std::endl;

	reduction<<<1, 4>>>(4, 0, 2, d_a);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaMemcpy(&a, d_a, 4*sizeof(float), cudaMemcpyDeviceToHost));
	for (int i = 0; i < 4; ++i) {
		std::cout << a[i] << " ";
	}
	std::cout << std::endl;
	cudaFree(d_a);

	/*
	 * Testing array size 5
	 */

	std::cout << std::endl << "Reduce array length 5" << std::endl;

	float b[] = {.25, .5, .75, 1, 1.25};
	float *d_b;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_b, 5*sizeof(float)));
	CUDA_CHECK_RETURN(cudaMemcpy(d_b, &b, 5*sizeof(float), cudaMemcpyHostToDevice));
	reduction<<<1, 5>>>(5, 0, 1, d_b);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaMemcpy(&b, d_b, 5*sizeof(float), cudaMemcpyDeviceToHost));
	for (int i = 0; i < 5; ++i) {
		std::cout << b[i] << " ";
	}
	std::cout << std::endl;

	reduction<<<1, 5>>>(5, 0, 2, d_b);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaMemcpy(&b, d_b, 5*sizeof(float), cudaMemcpyDeviceToHost));
	for (int i = 0; i < 5; ++i) {
		std::cout << b[i] << " ";
	}
	std::cout << std::endl;

	reduction<<<1, 5>>>(5, 0, 3, d_b);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaMemcpy(&b, d_b, 5*sizeof(float), cudaMemcpyDeviceToHost));
	for (int i = 0; i < 5; ++i) {
		std::cout << b[i] << " ";
	}
	std::cout << std::endl;

	cudaFree(d_b);


	/*
	 * Testing array size 7
	 */
	std::cout << std::endl << "Reduce array length 7" << std::endl;

	float c[] = {.25, .5, .75, 1, 1.25, 1.5, 1.75};
	float *d_c;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_c, 7*sizeof(float)));
	CUDA_CHECK_RETURN(cudaMemcpy(d_c, &c, 7*sizeof(float), cudaMemcpyHostToDevice));

	for (int j = 0; j < ceil(log2(7.0)); ++j) {
		reduction<<<1, 7>>>(7, 0, j+1, d_c);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		CUDA_CHECK_RETURN(cudaMemcpy(&c, d_c, 7*sizeof(float), cudaMemcpyDeviceToHost));
		for (int i = 0; i < 7; ++i) {
			std::cout << c[i] << " ";
		}
		std::cout << std::endl;
	}
	cudaFree(d_c);

	/*
	 * testing stacked arrays
	 */
	std::cout << std::endl << "testing stacked arrays 4x4" << std::endl;
	float d[] = {.25, .5, .75, 1, .1, .2, .3, .4, .2, .4, .6, .8, .3, .6, .9, 1.2};
	float *d_d;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_d, 16*sizeof(float)));
	CUDA_CHECK_RETURN(cudaMemcpy(d_d, &d, 16*sizeof(float), cudaMemcpyHostToDevice));
	for (int j = 0; j < ceil(log2(4.0)); ++j) {
		reduction<<<4, 4>>>(4, 4, j+1, d_d);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		CUDA_CHECK_RETURN(cudaMemcpy(&d, d_d, 16*sizeof(float), cudaMemcpyDeviceToHost));
		for (int i = 0; i < 16; ++i) {
			std::cout << d[i] << " ";
		}
		std::cout << std::endl;
	}
	cudaFree(d_d);


	/*
	 * testing stacked arrays
	 */
	std::cout << std::endl << "testing stacked arrays 4x5" << std::endl;
	float e[] = {.25, .5, .75, 1, 1.25, .1, .2, .3, .4, .5, .2, .4, .6, .8, 1, .3, .6, .9, 1.2, 1.5};
	float *d_e;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_e, 20*sizeof(float)));
	CUDA_CHECK_RETURN(cudaMemcpy(d_e, &e, 20*sizeof(float), cudaMemcpyHostToDevice));
	for (int j = 0; j < ceil(log2(5.0)); ++j) {
		reduction<<<4, 5>>>(5, 4, j+1, d_e);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		CUDA_CHECK_RETURN(cudaMemcpy(&e, d_e, 20*sizeof(float), cudaMemcpyDeviceToHost));
		for (int i = 0; i < 20; ++i) {
			std::cout << e[i] << " ";
		}
		std::cout << std::endl;
	}
	cudaFree(d_e);
}


size_t GPUNet::current_mem_usage(int dev) {
	return gpu_mem[dev];
}



/*
 * ------------ private ------------
 */


dim3 GPUNet::get_threadsm2l1() {
	dim3 threadsm2l1;
	int s = (int)ceil(sqrt(n_input+1));

	threadsm2l1.x = s;
	threadsm2l1.y = s;

	return threadsm2l1;
}

dim3 GPUNet::get_threadsm2l2() {
	dim3 threadsm2l2;
	int s = (int)ceil(sqrt(n_hidden+1));

	threadsm2l2.x = s;
	threadsm2l2.y = s;

	return threadsm2l2;
}

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
	return props.totalGlobalMem;
}


/*
 * Copies the host vector to a pointer array on the device
 * Cannot index from host
 */
void GPUNet::copy_to_device(thrust::host_vector<FeatureVector*> &hv, FeatureVector ***dv) {

	CUDA_CHECK_RETURN(cudaMalloc((void **)&(*dv), hv.size()*sizeof(FeatureVector*)));

	FeatureVector** host_dv_tmp = (FeatureVector**)malloc(hv.size()*sizeof(FeatureVector*));

	for (size_t i = 0; i < hv.size(); ++i) {
		//allocate device memory
		FeatureVector *d_fv;
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_fv, sizeof(FeatureVector)));

		float *d_inp, *d_tar;
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_inp, (n_input)*sizeof(float)));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_tar, (n_output)*sizeof(float)));

		CUDA_CHECK_RETURN(cudaMemcpy(&(d_fv->input), &d_inp, sizeof(float *), cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(d_inp, hv[i]->input, n_input*sizeof(float), cudaMemcpyHostToDevice));

		CUDA_CHECK_RETURN(cudaMemcpy(&(d_fv->target), &d_tar, sizeof(float *), cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(d_tar, hv[i]->target, n_output*sizeof(float), cudaMemcpyHostToDevice));

		for(int a = 0; a < n_input; a++) {
			std::cout << a << ": " << hv[i]->input[a] << std::endl;
		}

		print_floats<<<1,1>>>(n_input, d_inp, d_fv);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		std::cout << std::endl;

		host_dv_tmp[i] = d_fv;
	}

	std::cout << "precopy"<<std::endl;
    // Copy to device Memory
    CUDA_CHECK_RETURN(cudaMemcpy(*dv, host_dv_tmp, hv.size()*sizeof(FeatureVector*), cudaMemcpyHostToDevice));
    std::cout << "postcopy"<<std::endl;

	print_all<<<1,1>>>(n_input, n_output, *dv);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	std::cout << "Copying data 4" << std::endl;
}

/*
 * Copies the host vector to a pointer array on the device
 * Cannot index from host
 * The final bias node is included in the inputs. This is so that when referencing the inputs
 * 	the bias is not lost
 */
void GPUNet::copy_to_device_biased(thrust::host_vector<FeatureVector*> &hv, FeatureVector ***dv) {

	CUDA_CHECK_RETURN(cudaMalloc((void **)&(*dv), hv.size()*sizeof(FeatureVector*)));

	FeatureVector** host_dv_tmp = (FeatureVector**)malloc(hv.size()*sizeof(FeatureVector*));

	for (size_t i = 0; i < hv.size(); ++i) {
		//allocate device memory
		FeatureVector *d_fv;
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_fv, sizeof(FeatureVector)));

		float *d_inp, *d_tar;
		//allocate for bias
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_inp, (n_input+1)*sizeof(float)));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_tar, (n_output)*sizeof(float)));

		CUDA_CHECK_RETURN(cudaMemcpy(&(d_fv->input), &d_inp, sizeof(float *), cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(d_inp, hv[i]->input, (n_input)*sizeof(float), cudaMemcpyHostToDevice));

		CUDA_CHECK_RETURN(cudaMemcpy(&(d_fv->target), &d_tar, sizeof(float *), cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(d_tar, hv[i]->target, n_output*sizeof(float), cudaMemcpyHostToDevice));

		//TODO: does setting all in parallel improve speed?
		set_bias<<<1, 1>>>(n_input, d_inp);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		for(int a = 0; a < n_input; a++) {
			std::cout << a << ": " << hv[i]->input[a] << std::endl;
		}

		print_floats<<<1,1>>>(n_input+1, d_inp, d_fv);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		std::cout << std::endl;

		host_dv_tmp[i] = d_fv;
	}

	std::cout << "precopy"<<std::endl;
    // Copy to device Memory
    CUDA_CHECK_RETURN(cudaMemcpy(*dv, host_dv_tmp, hv.size()*sizeof(FeatureVector*), cudaMemcpyHostToDevice));
    std::cout << "postcopy"<<std::endl;

	print_all<<<1,1>>>(n_input, n_output, *dv);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	std::cout << "Copying data 4" << std::endl;
}

/*
 * Copies the host vector to a pointer array on the host that holds pointers to head FeatureVector on the device
 */
void GPUNet::copy_to_device_host_array(thrust::host_vector<FeatureVector*> &hv, FeatureVector ***dv) {

	(*dv) = (FeatureVector**)malloc(hv.size()*sizeof(FeatureVector));
	//FeatureVector** host_dv_tmp = (FeatureVector**)malloc(hv.size()*sizeof(FeatureVector*));

	for (size_t i = 0; i < hv.size(); ++i) {
		//allocate device memory
		FeatureVector *d_fv;
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_fv, sizeof(FeatureVector)));

		float *d_inp, *d_tar;
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_inp, (n_input)*sizeof(float)));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_tar, (n_output)*sizeof(float)));

		CUDA_CHECK_RETURN(cudaMemcpy(&(d_fv->input), &d_inp, sizeof(float *), cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(d_inp, hv[i]->input, n_input*sizeof(float), cudaMemcpyHostToDevice));

		CUDA_CHECK_RETURN(cudaMemcpy(&(d_fv->target), &d_tar, sizeof(float *), cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(d_tar, hv[i]->target, n_output*sizeof(float), cudaMemcpyHostToDevice));



		//for(int a = 0; a < n_input; a++) {
		//	std::cout << a << ": " << hv[i]->input[a] << std::endl;
		//}

		//print_floats<<<1,1>>>(n_input, d_inp, d_fv);
		//CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		//std::cout << std::endl;

		(*dv)[i] = d_fv;
	}

	//for (int i = 0; i < 4; ++i) {
	//	print_floats2<<<1,1>>>(n_input, (*dv)[i]);
	//	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	//}

}

/*
 * Copies the host vector to a pointer array on the host that holds pointers to FeatureVector on the device with bias node
 */
void GPUNet::copy_to_device_host_array_biased(thrust::host_vector<FeatureVector*> &hv, FeatureVector ***dv) {

	(*dv) = (FeatureVector**)malloc(hv.size()*sizeof(FeatureVector));
	//FeatureVector** host_dv_tmp = (FeatureVector**)malloc(hv.size()*sizeof(FeatureVector*));

	for (size_t i = 0; i < hv.size(); ++i) {
		//allocate device memory
		FeatureVector *d_fv;
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_fv, sizeof(FeatureVector)));

		float *d_inp, *d_tar;
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_inp, (n_input+1)*sizeof(float)));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_tar, (n_output)*sizeof(float)));

		CUDA_CHECK_RETURN(cudaMemcpy(&(d_fv->input), &d_inp, sizeof(float *), cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(d_inp, hv[i]->input, n_input*sizeof(float), cudaMemcpyHostToDevice));

		CUDA_CHECK_RETURN(cudaMemcpy(&(d_fv->target), &d_tar, sizeof(float *), cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(d_tar, hv[i]->target, n_output*sizeof(float), cudaMemcpyHostToDevice));


		//TODO: does setting all in parallel improve speed?
		set_bias<<<1, 1>>>(n_input, d_inp);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		for(int a = 0; a < n_input; a++) {
			std::cout << a << ": " << hv[i]->input[a] << std::endl;
		}

		print_floats<<<1,1>>>(n_input+1, d_inp, d_fv);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		std::cout << std::endl;

		//for(int a = 0; a < n_input; a++) {
		//	std::cout << a << ": " << hv[i]->input[a] << std::endl;
		//}

		//print_floats<<<1,1>>>(n_input, d_inp, d_fv);
		//CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		//std::cout << std::endl;

		(*dv)[i] = d_fv;
	}

	//for (int i = 0; i < 4; ++i) {
	//	print_floats2<<<1,1>>>(n_input, (*dv)[i]);
	//	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	//}
}


/*
 * Copies the host vector to a pointer array on the host that holds pointers to FeatureVector on the device with bias node
 */
void GPUNet::copy_to_device_host_array_ptrs_biased(thrust::host_vector<FeatureVector*> &hv, FeatureVector ***dv) {

	*dv = (FeatureVector**)malloc(hv.size()*sizeof(FeatureVector*));
	//FeatureVector** host_dv_tmp = (FeatureVector**)malloc(hv.size()*sizeof(FeatureVector*));

	for (size_t i = 0; i < hv.size(); ++i) {
		//allocate device memory
		FeatureVector *d_fv = (FeatureVector*)malloc(sizeof(FeatureVector*));

		float *d_inp, *d_tar;
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_inp, (n_input+1)*sizeof(float)));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_tar, (n_output)*sizeof(float)));

		CUDA_CHECK_RETURN(cudaMemcpy(d_inp, hv[i]->input, n_input*sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(d_tar, hv[i]->target, n_output*sizeof(float), cudaMemcpyHostToDevice));

		d_fv->input = d_inp;
		d_fv->target = d_tar;

		//TODO: does setting all in parallel improve speed?
		set_bias<<<1, 1>>>(n_input, d_inp);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		(*dv)[i] = d_fv;
	}

}
