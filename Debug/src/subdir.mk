################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/Net.cpp \
../src/NetData.cpp \
../src/NetTrainer.cpp \
../src/Profiler.cpp 

CU_SRCS += \
../src/GPUNet.cu \
../src/ann.cu 

CU_DEPS += \
./src/GPUNet.d \
./src/ann.d 

OBJS += \
./src/GPUNet.o \
./src/Net.o \
./src/NetData.o \
./src/NetTrainer.o \
./src/Profiler.o \
./src/ann.o 

CPP_DEPS += \
./src/Net.d \
./src/NetData.d \
./src/NetTrainer.d \
./src/Profiler.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-5.5/bin/nvcc -G -g -O0 -gencode arch=compute_30,code=sm_30 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-5.5/bin/nvcc --device-c -G -O0 -g -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-5.5/bin/nvcc -G -g -O0 -gencode arch=compute_30,code=sm_30 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-5.5/bin/nvcc -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


