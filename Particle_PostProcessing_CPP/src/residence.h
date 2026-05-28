#ifndef CLASS_RESIDENCE
#define CLASS_RESIDENCE

#include <iostream>

#include "stParticle.h"
#include "stSetting.h"

class ParticleResidenceTime {
	private:
		int*						n_residence_map;
		float*					residence_particle;
		float*					residence_map;

	public:
		void		allocate_residence(Setting& setting);
		void		cal_residence(int t, std::vector<Particle_Index> p_resid_odd, std::vector<Particle_Index> p_resid_even, Setting& setting);
		void		output_residence(Setting& setting);
		void		delete_residence();
};

#endif
