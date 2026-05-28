#include <iostream>
#include <vector>

#include "stSetting.h"
#include "stParticle.h"

class ParticleFlux {
	private:
		int*		flux;

	public:
		int*		num_p_flux;

		void		allocate_flux(Setting& setting);
		void		cal_flux(int t, std::vector<Particle_Index>& p_flux_odd, std::vector<Particle_Index>& p_flux_even, Setting& setting);
		void		output_flux(Setting& setting);
		void	  delete_flux();
};

