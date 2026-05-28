#ifndef CLASS_SETTING
#define CLASS_SETTING

#include <iostream>

class Set_Value {
	public:
		// Read
		void 		option_read(char* argv[], Setting& setting);

		// Output to console
		void		output_info();
		void		output_setting(Setting& setting);

		// Delete array
		void		delete_array (Setting setting) {
							delete[] setting.Z_OUT;
							delete[] setting.CTR_SENSOR;
						};
};

#endif
