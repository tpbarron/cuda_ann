#include <iostream>
#include <cstdio>
#include <time.h>
#include "NetData.h"
#include "Net.h"
#include "GPUNet.h"
#include "GPUNetSettings.h"
#include "NetTrainer.h"
#include "Profiler.h"
#include <boost/lexical_cast.hpp>

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	}																		\
}


/**
 * Profile network
 */
void profile(GPUNet &gnet, Net &net, NetData &d) {
	NetTrainer nt(&net);
	Profiler p(&gnet, &net, &nt);
	p.set_iterations(100);
	p.profile_feed_forward_v1_2(d);
	p.profile_feed_forward_v1_3(d);
	//p.profile_feed_forward_v2_2(d);
	//p.profile_cpu_feedforward(d.get_training_dataset()->training_set[0]->input);

	//p.profile_backprop_v2(d);
	//p.profile_cpu_backprop(d.get_training_dataset()->training_set[0]->target);
}

/**
 * Test network
 */
void test(GPUNet &gnet, Net &net, NetData &d) {
	gnet.init_from_net(net, d);
	gnet.test_feed_forward(net, d);
	gnet.test_backprop(net, d);
	//gnet.test_reduction();
}


int main(void) {
	srand(time(NULL));

	time_t start, stop;

	NetData d;
	if (!d.load_file("datasets/face.dat.norm"))
	//if (!d.load_file("datasets/and.dat"))
		return 0; //if file did not load
	//d.print_loaded_patterns();

//	Net net(d.num_inputs(), ceil(2.0/3.0*d.num_inputs()), d.num_targets());
//	GPUNet gnet(d.num_inputs(), d.num_targets(), GPUNetSettings::STANDARD);
//
//	gnet.alloc_dev_mem();
//	gnet.init_from_net(net, d);

//	gnet.init_net();
//	gnet.print_net();
//	std::cout << "Dev 0: " << gnet.current_mem_usage(0) << std::endl;

//	test(gnet, net, d);
//	profile(gnet, net, d);
//	gnet.run_parallel(net, d);
//	return 0;

	GPUNet gnet("nets/face.net");

	gnet.set_save_frequency(1);
	gnet.set_training_params(0.9, 0.9, false);
	gnet.set_stopping_conds(31, 95.0);
	start = clock();
	gnet.train_net_sectioned(d.get_training_dataset());
	stop = clock();
	std::cout << "GPU time: " << ((float)stop - start) / CLOCKS_PER_SEC << std::endl;
	//gnet.print_net();
	//gnet.write_net("nets/face_final.net");


//	NetTrainer nt(&net);
//	nt.set_stopping_conds(10000, 95);
//	nt.set_training_params(.9, .9);
//	start = clock();
//	nt.train_net(d.get_training_dataset());
//	stop = clock();
//	std::cout << "CPU time: " << ((double)stop - start) / CLOCKS_PER_SEC << std::endl;

	CUDA_CHECK_RETURN(cudaDeviceReset());
	std::cout << "Device reset" << std::endl;

	return 0;
}
