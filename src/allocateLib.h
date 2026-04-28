#ifndef ALLOCATELIB_H_
#define ALLOCATELIB_H_


#include <iostream>
#include <cuda.h>
#include <cstdio>
//#include <cutil.h>

#include "macroCUDA.h"


namespace	
allocateLib {
	// template *****
	// allocate
	template <typename T> 
	void 
	new_host   (T **f_h, int n)
	{
		*f_h = (T *) malloc (sizeof(T) * n);
	}


	template <typename T> 
	void 
	new_device (T **f_d, int n)
	{
		CUDA_SAFE_CALL( cudaMalloc((void **)f_d, sizeof(T) * n) );
	}


	template <typename T> 
	void 
	new_pinned(T **f_h, int n)
	{
		CUDA_SAFE_CALL( cudaMallocHost((void **)f_h, sizeof(T) * n) );
	}


	// host + device
	template <typename T> 
	void
	new_host_device (T **f_h, T **f_d, int n)
	{
		new_host   (f_h, n);
		new_device (f_d, n);
	}


	// host(pinned) + device
	template <typename T> 
	void
	new_pind_device (T **f_h, T **f_d, int n)
	{
		new_pinned (f_h, n);
		new_device (f_d, n);
	}

}


#endif
