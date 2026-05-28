#ifndef CLASS_DENSITY
#define CLASS_DENSITY

#include <iostream>

#include "stSetting.h"

class ParticleDensity {
	private:
		int*		xy_number_density;
		int*		xz_number_density;
		int*		yz_number_density;
	public:
		void		allocate_density(Setting& setting);
		void		cal_density(float x, float y, float z, Setting& setting);
		void		output_density(Setting& setting);
		void		delete_density();
};

#endif
