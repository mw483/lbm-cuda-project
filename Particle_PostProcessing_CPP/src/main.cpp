#include <iostream>
#include <time.h>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <stdlib.h>

#include "stSetting.h"
#include "function.h"
#include "setting.h"
#include "calculation.h"

/* -------------------------------------     Main        ----------------------------------------------*/ 

int main(int argc, char *argv[]) {
	// Struct
	Setting					setting;

	// Class
	Set_Value				set_value;

	set_value.output_info();

	if (argc >= 2) {
		set_value.option_read(argv, setting);
		set_value.output_setting(setting);
	}
	
	std::cout << "Start post-processing. " << std::endl;
	std::cout << std::endl;
		
	if (setting.FLG_FOOT == 1 || setting.FLG_FLUX == 1 || setting.FLG_RESID) {
		calculation_index(setting);
	} else {
		calculation_pos(setting);
	}
	
	set_value.delete_array(setting);
	return 0;
}

