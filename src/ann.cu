#include <cstdio>
#include <iostream>
#include <time.h>

#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>

#include "GPUNet.h"
#include "GPUNetSettings.h"
#include "Net.h"
#include "NetData.h"
#include "NetTrainer.h"
#include "Profiler.h"


namespace boost_po = boost::program_options;

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
	p.profile_feed_forward_v2(d);
	p.profile_cpu_feedforward(&d.get_training_dataset()->training_set[0]);

	p.profile_backprop_v2(d);
	p.profile_backprop_v3(d);
	p.profile_cpu_backprop(&d.get_training_dataset()->training_set[d.get_training_dataset()->n_input+1]);
}

/**
 * Test network
 */
void test(GPUNet &gnet, Net &net, NetData &d) {
	gnet.init_from_net(net, d);
	gnet.test_feed_forward(net, d);
	gnet.test_backprop(net, d);
}

int main(int argc, char **argv) {
	srand(time(NULL));
	time_t start, stop;

	float l_rate, momentum, t_set_pct, hidden_pct;
	int max_epochs, save_freq;
	std::string dset, netf, fbase;

	boost_po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message")
		("dataset,d", boost_po::value<std::string>(&dset), "data set file")
		("loadnet,n", boost_po::value<std::string>(&netf), "load net file")
		("profile,p", boost_po::bool_switch(), "profile GPU functions")
		("validate,v", boost_po::bool_switch(), "validate GPU functions")
		("test,t", boost_po::bool_switch(), "run test set, this will take a different random sampling to be the test set on every initialization")
		("f_base,f", boost_po::value<std::string>(&fbase)->default_value("itr"), "base name of net file when writing, default = [itr]_#.txt")
		("l_rate,r", boost_po::value<float>(&l_rate)->default_value(0.7), "learning rate, default = 0.7")
		("hidden_pct,h", boost_po::value<float>(&hidden_pct)->default_value(2.0/3.0), "number of hidden nodes as percentage input nodes, default = 2.0/3.0")
		("t_pct,c", boost_po::value<float>(&t_set_pct)->default_value(0.8), "percentage of dataset used for training, default = 0.8")
		("momentum,m", boost_po::value<float>(&momentum)->default_value(0.9), "momentum, default = 0.9")
		("batch,b", boost_po::bool_switch()->default_value(false), "batch update, default = 0 (false), will ignore momentum")
		("max_epochs,e", boost_po::value<int>(&max_epochs)->default_value(1000), "max epochs, default = 1000")
		("save_freq,s", boost_po::value<int>(&save_freq)->default_value(100), "save data every n epochs, default = 100")
		("cpu", boost_po::bool_switch(), "run on CPU instead of GPU")
		("reset", boost_po::bool_switch(), "reset all CUDA capable GPUs")
		("parallel", boost_po::bool_switch(), "Run networks in parallel on CPU and GPU to compare")
	;
	boost_po::positional_options_description p;
	p.add("dataset", -1);


	boost_po::variables_map vm;
	boost_po::store(boost_po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
	boost_po::notify(vm);

	if (!vm.size()) {
		std::cout << "Try: cuda_ann --help\n\n";
		return 1;
	}

	if (vm.count("help")) {
		std::cout << desc << std::endl;;
		return 1;
	}

	if (vm["reset"].as<bool>()) {
		int count;
		CUDA_CHECK_RETURN(cudaGetDeviceCount(&count));
		for (int i = 0; i < count; i++) {
			std::cout << "Resetting gpu: " << i << std::endl;
			CUDA_CHECK_RETURN(cudaSetDevice(i));
			CUDA_CHECK_RETURN(cudaDeviceReset());
		}
		return 1;
	}

	if (!vm.count("dataset")) {
		std::cerr << "Must have dataset parameter" << std::endl;
		return 1;
	}

	NetData d(t_set_pct);
	if (!d.load_file(dset))
		return 1; //if file did not load
	//d.print_loaded_patterns_flatted();

	bool net_loaded = false;
	GPUNet gnet;
	Net net;
	if (vm.count("loadnet")) {
		std::cout << "Using netfile to initialize" << std::endl;
		net_loaded = gnet.load_netfile(netf);
	} else {
		//init normally
		std::cout << "Using " << hidden_pct << " * " << "n_input for hidden nodes" << std::endl;
		net.init(d.num_inputs(), d.num_targets(), hidden_pct);
		gnet.init(d.num_inputs(), d.num_targets(), hidden_pct, GPUNetSettings::STANDARD);
		gnet.alloc_host_mem();
		gnet.alloc_dev_mem();
		gnet.init_from_net(net, d);
	}

//	FeatureVector **dv;
//	thrust::host_vector<FeatureVector*> hv;
//	for (int i = 0; i < d.data.size(); ++i) {
//		hv.push_back(d.data[i]);
//	}
//	gnet.copy_to_device_host_array_ptrs_biased(hv,&dv);
//
//	float*d_set;
//	gnet.copy_to_device(d.get_training_dataset()->training_set, d.get_training_dataset()->n_training, d.get_training_dataset()->fpp, &d_set);
//
//	return 0;

	bool train = true;
	if (vm["validate"].as<bool>()) {
		test(gnet, net, d);
		train = false;
	}
	if (vm["profile"].as<bool>()) {
		profile(gnet, net, d);
		train = false;
	}
	if (vm["parallel"].as<bool>()) {
		gnet.run_parallel(net, d);
		train = false;
	}

	if (train) {
		bool batching = vm["batch"].as<bool>();
		if (batching)
			std::cout << "Using batch learning mode" << std::endl;
		else
			std::cout << "Using stochastic learning mode" << std::endl;

		if (vm["cpu"].as<bool>()) {
			std::cout << "CPU flag set" << std::endl;
			NetTrainer nt = NetTrainer(&net);
			nt.set_stopping_conds(max_epochs, 97.5);
			nt.set_training_params(l_rate, momentum, batching);
			start = clock();
			nt.train_net(d.get_training_dataset());
			stop = clock();
			std::cout << "CPU time: " << ((float)stop - start) / CLOCKS_PER_SEC << std::endl;
		} else {
			if (vm["test"].as<bool>()) {
				gnet.run_test_set(d.get_training_dataset());
			} else {
				gnet.set_base_file_name(fbase);
				gnet.set_save_frequency(save_freq);
				gnet.set_training_params(l_rate, momentum, batching);
				gnet.set_stopping_conds(max_epochs, 95.0);
				start = clock();
				gnet.train_net_sectioned_overlap(d.get_training_dataset());
				stop = clock();
				std::cout << "GPU time: " << ((float)stop - start) / CLOCKS_PER_SEC << std::endl;
			}
		}
	}

//	CUDA_CHECK_RETURN(cudaDeviceReset());
//	std::cout << "Device reset" << std::endl;

	return 0;
}
