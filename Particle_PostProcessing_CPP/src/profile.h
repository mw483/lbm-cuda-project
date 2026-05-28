#ifndef CLASS_PROFILE
#define CLASS_PROFILE

#include <iostream>

#include "stSetting.h"

class ParticleProfile {
	private:
		int*		profile;
		int*		fluid_grid;

	public:
		void		allocate_profile(Setting& setting);
		void		read_map(Setting& setting);
		void		cal_profile(float x, float y, float z, Setting& setting);
		void		output_profile(Setting& setting);
		void		delete_profile();
};

#endif
