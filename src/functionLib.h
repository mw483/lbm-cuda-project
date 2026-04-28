#ifndef FUNCTIONLIB_H_
#define FUNCTIONLIB_H_

#include "paramDomain.h"
#include <iostream>
#include "definePrecision.h"

namespace	
functionLib {

	// template *****
	template <typename T>
	void 
	swap (T **f, T **fn)
	{
		T	*tmp;
	
		tmp = *f;
		*f  = *fn;
		*fn = tmp;
	}


	template <typename T1, typename T2> 
	void
	fillArray (T1 *f, T2 a, int n)
	{
		for (int i=0; i<n; i++) { f[i] = (T1)a; }
	}



	template <typename T1, typename T2>
	void
	fillArrayF0 (T1 *f, T2 a, FLOAT dTdz0, FLOAT dTdz1, int kzi, int nx, int ny, int nz, FLOAT dx)
	{
		int	id;
		int	nxy=nx*ny;
		int	nxyz=nx*ny*nz;
		T1	rho;
		T1	rho_air=1.0;
		T1	dTdz;
		for (int k=0; k<kzi; k++) {
			dTdz = (k<kzi) ? (T1)dTdz0 : (T1)dTdz1;
			rho = rho_air * ((T1)1.0 + (T1)k * (T1)dTdz * (T1)dx / (T1)a);
			for (int j=0; j<ny; j++) {
				for (int i=0; i<nx; i++) {
					for (int n=0; n<27; n++) {
						id = i + j * nx + k * nxy + n * nxyz;
						f[id] = (T1)rho;
					}
				}
			}
		}
	}



	template <typename T1, typename T2>
	void
	fillArrayT0 (T1 *f, T2 a, FLOAT dtdz0, FLOAT dtdz1, int kzi, int nx, int ny, int nz, FLOAT dx)
	{
//		const FLOAT	dzdT0 =	user_init::DTDZ_LOW;	// 0.0; // K/m   default: 0.01
//		const FLOAT	dzdT1 = user_init::DTDZ_HIGH;	// 0.01; // K/m   default: 0.01
//		const int	kzi = user_init::ZHIGH/dx;	// 200; // zi/dx
		int	id;
		T1	TEMP;
		int	nxy = nx * ny;

//		std::cout << "--Vertical temperature gradient--" << std::endl;
//		std::cout << "dTdz="<<dtdz0<<"  k<"<<kzi<< std::endl;
//		std::cout << "dTdz="<<dtdz1<<"  k>"<<kzi<< std::endl;

// within ABL
		TEMP = (T1)a;
		for (int k=0; k<kzi; k++) {
			for (int j=0; j<ny; j++) {
				for (int i=0; i<nx; i++) {
					id = i + j * nx + k * nxy;
					f[id] = TEMP + (T1)k * (T1)dtdz0 * (T1)dx;
				}
			}
		}
// above ABL (free atmospehre)
		TEMP = (T1)a + kzi * dtdz0 * dx;
		for (int k=kzi; k<nz; k++) {
			for (int j=0; j<ny; j++) {
				for (int i=0; i<nx; i++) {
					id = i + j * nx + k * nxy;
					f[id] = TEMP + ((T1)k-(T1)kzi) * (T1)dtdz1 * (T1)dx;
				}
			}
		}
//		TEMP =  (T1)a+ kzi * dtdz0 * dx + ((T1)kz2-(T1)kzi) * (T1)dtdz1 * (T1)dx;
//                for (int k=kz2; k<nz; k++) {
//                        for (int j=0; j<ny; j++) {
//                                for (int i=0; i<nx; i++) {
//                                        id = i + j * nx + k * nxy;
//                                        f[id] = TEMP;
//                                }
//                        }
//                }

	}



	template <typename T> 
	void
	addArray (int n, T *c, const T *a, const T *b)
	{
		for (int i=0; i<n; i++) {
			c[i] = a[i] + b[i];
		}
	}
	

	template <typename T> 
	void
	diffArray(int n, T *c, const T *a, const T *b)
	{
		for (int i=0; i<n; i++) {
			c[i] = a[i] - b[i];
		}
	}


	template <typename T> 
	void
	init2 (T *f, T a, T b)
	{
		f[0] = a;
		f[1] = b;
	}


	template <typename T> 
	void
	init3 (T *f, T a, T b, T c)
	{
		f[0] = a;
		f[1] = b;
		f[2] = c;
	}


	void
	set_dim3 (dim3 *dim_cuda, int dim_x, int dim_y, int dim_z);


	// 値をMの倍数の小さい側の整数にキャストする(f=m なら return m)
	template <typename T> 
	int
	cast_value_small (T f, unsigned int m)
	{
		if (f < (T)0) { 
			std::cout << "error cast_value_less\n";
			std::cout << "      f < 0          \n";
	
			exit(-1);
		}
	
		return	((int)(f/m)) * m;
	}
	
	
	// 値をMの倍数の大きい側の整数にキャストする(f=m なら return m)
	template <typename T> 
	int
	cast_value_large (T f, unsigned int m)
	{
		if (f < (T)0) { 
			std::cout << "error cast_value_less\n";
			std::cout << "      f < 0          \n";
	
			exit(-1);
		}
	
		return	((int)((f+m-1)/m)) * m;
	}
	
	FLOAT
	get_elapsed_time (
		const struct timeval *begin,
		const struct timeval *end);


	// inc //
}


#include "functionLib_inc.h"


#endif
