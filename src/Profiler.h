/*
 * Profiler.h
 *
 *  Created on: Feb 19, 2014
 *      Author: trevor
 */

#ifndef PROFILER_H_
#define PROFILER_H_

#include <time.h>
#include "GPUNet.h"
#include "Net.h"
#include "NetTrainer.h"

const int DEFAULT_ITERATIONS = 1000;

class Profiler {
public:
	Profiler(GPUNet *gnet, Net *net, NetTrainer *nt);
	~Profiler();

	void set_iterations(int i);
	int get_iterations();

	double profile_feed_forward_v1();
	double profile_feed_forward_v1_2();
	double profile_feed_forward_v2();
	double profile_feed_forward_v2_2();

	double profile_backprop_v1();
	double profile_backprop_v2();
	double profile_cpu_backprop(float *targets);
	double profile_cpu_feedforward(float *targets);
	double profile_mse_acc();
	double profile_weight_init();


private:

	GPUNet *gnet;
	Net *net;
	NetTrainer *nt;
	int iterations;
	clock_t start, stop;

};

#endif /* PROFILER_H_ */
