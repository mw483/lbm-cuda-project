#ifndef PARAMCALFLAG_H_
#define PARAMCALFLAG_H_


#include "stCalFlag.h"


class	
paramCalFlag {
private:	
	// stCalFlag.h //
	CalFlag		calflg_;

public:	
	paramCalFlag () {}

	paramCalFlag (
		char	*program_name,
		int		argc,
	   	char	*argv[])
	{
		set (program_name, argc, argv);
	}
	
	~paramCalFlag () {}


	// CalFlag //
	int		restart()		const { return	calflg_.restart; }

	int		fstart()		const { return	calflg_.fstart; }
	int		pstartstep()	const { return	calflg_.pstartstep; }

	int		cout_step()		const { return	calflg_.cout_step; }
	int		fout_step()		const { return	calflg_.fout_step; }

	int		cout_flg()		const { return	calflg_.cout_frg; }
	int		fout_flg()		const { return	calflg_.fout_frg; }

	int		restart_out_flg()		const { return	calflg_.restart_out_frg; }

	const CalFlag	&calflg() const { return calflg_; }

public:	
	void
	set (
		char	*program_name,
		int		argc,
	   	char	*argv[]);


	// cout
	void	cout_CalFlag ();
	void	cout_CalFlag (const CalFlag	&calflg);
	
};


#endif
