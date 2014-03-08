/*
 * GPUNetTests.cpp
 *
 *  Created on: Mar 5, 2014
 *      Author: trevor
 */

#define BOOST_TEST_DYN_LINK

#define BOOST_TEST_MODULE example
#include <boost/test/unit_test.hpp>
#include "../src/NetData.h"
#include "../src/Net.h"
#include "../src/GPUNet.h"
#include "../src/NetTrainer.h"
//#include "../src/Profiler.h"

#include <iostream>
#include <cstdio>
#include <time.h>


struct F {
    F() : i( 0 ) { BOOST_TEST_MESSAGE( "setup fixture" ); }
    ~F()         { BOOST_TEST_MESSAGE( "teardown fixture" ); }

    int i;
};

struct GpuF {

	NetData d;
	GPUNet gnet;

	float *d_input;
	float *d_output;

	GpuF() {
		//NetData d;
		if (d.load_file("datasets/and.dat"))
			exit(0); //if file did not load
		//d.print_loaded_patterns();

		Net net(d.num_inputs(), ceil(2.0/3.0*d.num_inputs()), d.num_targets());
		gnet.init_structure(d.num_inputs(), d.num_targets(), GPUNet::STANDARD);

		clock_t start, finish;

		for (int i = 0; i < net.n_input; ++i) {
			std::cout << "i = " << i << std::endl;
			std::cout << d.get_training_dataset()->training_set[0]->input[i] << std::endl;
		}
		std::cout << std::endl;

		std::cout << "feed forward CPU" << std::endl;
		start = clock();
		net.feed_forward(d.get_training_dataset()->training_set[0]->input);
		finish = clock();
		std::cout << "feed forward CPU time: " << ((float)(finish-start)) / CLOCKS_PER_SEC << "s\n\n";
	}

	~GpuF() {}

};



/*
 * Test feed forward
 */

BOOST_FIXTURE_TEST_SUITE( s, GpuF )

BOOST_AUTO_TEST_CASE( test_feed_forward_v1 )
{
	std::cout << "Testing method 1" << std::endl;
	gnet.feed_forward_v1();
	std::cout << "Validates: " << gnet.validate_output(net.outputNeurons) << "\n";
	CUDA_CHECK_RETURN(cudaMemset(d_output, 0, n_output*sizeof(float)));

    BOOST_CHECK( i == 1 );
}


BOOST_AUTO_TEST_CASE( test_feed_forward_v1_2 )
{

	std::cout << "Testing method 1.2" << std::endl;
	FeatureVector **dv;
	gnet.copy_to_device_host_array_ptrs_biased(d.get_training_dataset()->training_set, &dv);
	gnet.feed_forward_v1_2(dv[0]->input);
	std::cout << "Validates: " << gnet.validate_output(net.outputNeurons) << "\n";
	CUDA_CHECK_RETURN(cudaMemset(d_output, 0, n_output*sizeof(float)));

    BOOST_CHECK_EQUAL( i, 0 );
}

BOOST_AUTO_TEST_CASE( test_feed_forward_v2 )
{
	std::cout << "Testing method 2" << std::endl;
	feed_forward_v2();
	std::cout << "Validates: " << validates(net.outputNeurons) << "\n";
	CUDA_CHECK_RETURN(cudaMemset(d_output, 0, n_output*sizeof(float)));
}


BOOST_AUTO_TEST_CASE( test_feed_forward_v2_2 )
{
	std::cout << "Testing method 2.2" << std::endl;
	feed_forward_v2_2();
	std::cout << "Validates: " << validates(net.outputNeurons) << "\n";
	CUDA_CHECK_RETURN(cudaMemset(d_output, 0, n_output*sizeof(float)));
}


BOOST_AUTO_TEST_SUITE_END()
