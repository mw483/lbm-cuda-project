#ifndef STCALFLAG_H_
#define STCALFLAG_H_


struct
CalFlag {
	int		restart; // 0 : init, 1 : restart

	int		fstart;
	int		pstartstep;

	int		cout_step, fout_step;

	int		cout_frg, fout_frg;
	int		restart_out_frg;
};


#endif
