#ifndef CLASS_NUM
#define CLASS_NUM

#include <iostream>

#include "stSetting.h"

class ParticleNum {
	private:
		int*	total_num;
	public:
		void	allocate_num(Setting& setting);
		void	cal_num(int t, int n, Setting& setting);
		void	output_num(Setting& setting);
		void 	delete_num();
	
};

#endif
