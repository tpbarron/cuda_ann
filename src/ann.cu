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
#include <boost/program_options.hpp>

namespace po = boost::program_options;

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
	//p.profile_feed_forward_v1_3(d);
	//p.profile_feed_forward_v2_2(d);
	//p.profile_cpu_feedforward(d.get_training_dataset()->training_set[0]->input);

	p.profile_backprop_v2(d);
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

int main(int argc, char **argv) {
	srand(time(NULL));
	time_t start, stop;

	float l_rate, momentum, t_set_pct;
	int max_epochs, save_freq;
	std::string dset, netf, fbase;

	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message")
		("dataset,d", po::value<std::string>(&dset), "data set file")
		("loadnet,n", po::value<std::string>(&netf), "load net file")
		("profile,p", po::bool_switch(), "profile GPU functions")
		("test,t", po::bool_switch(), "test GPU functions")
		("fbase,f", po::value<std::string>(&fbase)->default_value("itr"), "base name of net file when writing, default = [itr]_#.txt")
		("l_rate,r", po::value<float>(&l_rate)->default_value(0.7), "learning rate, default = 0.7")
		("t_pct,c", po::value<float>(&t_set_pct)->default_value(0.8), "percentage of dataset used for training, default = 0.8")
		("momentum,m", po::value<float>(&momentum)->default_value(0.9), "momentum, default = 0.9")
		("batch,b", po::bool_switch()->default_value(false), "batch update, default = 0 (false), will ignore momentum")
		("max_epochs,e", po::value<int>(&max_epochs)->default_value(1000), "max epochs, default = 1000")
		("save_freq,s", po::value<int>(&save_freq)->default_value(5), "save data every n epochs, default = 5")
		("cpu", po::bool_switch(), "run on CPU instead of GPU")
	;
	po::positional_options_description p;
	p.add("dataset", -1);

	po::variables_map vm;
	po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
	po::notify(vm);

	if (!vm.size()) {
		std::cout << "Try: cuda_ann --help\n\n";
		return 1;
	}

	if (vm.count("help")) {
		std::cout << desc << std::endl;;
		return 1;
	}

	//std::cout << dset << " " << netf << " " << l_rate << " " << momentum << " " << max_epochs << " " << save_freq << " " << batch << std::endl;
	if (!vm.count("dataset")) {
		std::cerr << "Must have dataset parameter" << std::endl;
		return 1;
	}

	NetData d(t_set_pct);
	if (!d.load_file(dset))
		return 1; //if file did not load
	//d.print_loaded_patterns_flatted();

	GPUNet gnet;
	Net net;
	if (vm.count("loadnet")) {
		std::cout << "Using netfile to initialize" << std::endl;
		gnet.load_netfile(netf);
	} else {
		//init normally
		net.init(d.num_inputs(), ceil(1.0/3.0*d.num_inputs()), d.num_targets());
		gnet.init(d.num_inputs(), d.num_targets(), GPUNetSettings::STANDARD);
		gnet.alloc_host_mem();
		gnet.alloc_dev_mem();
		gnet.init_from_net(net, d);
	}

	bool train = true;
	if (vm["test"].as<bool>()) {
		std::cout << "testing" <<std::endl;
		test(gnet, net, d);
		train = false;
	}
	if (vm["profile"].as<bool>()) {
		std::cout << "profiling" << std::endl;
		profile(gnet, net, d);
		train = false;
	}
	if (train) {
		if (vm["cpu"].as<bool>()) {
			std::cout << "--cpu set" << std::endl;
			NetTrainer nt = NetTrainer(&net);
			nt.set_stopping_conds(max_epochs, 97.5);
			nt.set_training_params(l_rate, momentum);
			start = clock();
			nt.train_net(d.get_training_dataset());
			stop = clock();
			std::cout << "CPU time: " << ((float)stop - start) / CLOCKS_PER_SEC << std::endl;
		} else {
			gnet.set_base_file_name(fbase);
			gnet.set_save_frequency(save_freq);
			gnet.set_training_params(l_rate, momentum, vm["batch"].as<bool>());
			gnet.set_stopping_conds(max_epochs, 100.0);
			start = clock();
			gnet.train_net_sectioned(d.get_training_dataset());
			stop = clock();
			std::cout << "GPU time: " << ((float)stop - start) / CLOCKS_PER_SEC << std::endl;
		}
	}

//	CUDA_CHECK_RETURN(cudaDeviceReset());
//	std::cout << "Device reset" << std::endl;

	return 0;
}
