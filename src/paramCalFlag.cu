#include "paramCalFlag.h"

#include <iostream>
#include "option_parser.h"


// public //
void	paramCalFlag::
set (
	char	*program_name,
	int		argc,
	char	*argv[])
{
	const char	program_args[] = "[options...]";
	OptionParser parser(program_name, program_args);

	// This function must be executed before other functions are called.
	const int	ret = parser.parse_args(argc, argv);
	if (ret != 1)						{ exit(2); }
	if (! parser.check_narguments(0))	{ exit(0); }


	// CalFlag //
	calflg_.restart = parser.restart();

	calflg_.fstart     = parser.fstart();
	calflg_.pstartstep = parser.pstartstep();

	calflg_.cout_step = parser.cfout(0);
	calflg_.fout_step = parser.cfout(1);

	calflg_.cout_frg        = parser.cfrfrg(0);
	calflg_.fout_frg        = parser.cfrfrg(1);
	calflg_.restart_out_frg = parser.cfrfrg(2);
}


// cout
void	paramCalFlag::
cout_CalFlag ()
{
	std::cout << "// cout_CalFlag //" << std::endl;
	std::cout << "restart            = " << calflg_.restart << std::endl;
	std::cout << "cout, fout  (step) = " << calflg_.cout_step << ", " << calflg_.fout_step << std::endl;
	std::cout << "cout, fout  (flag) = " << calflg_.cout_frg  << ", " << calflg_.fout_frg  << std::endl;
	std::cout << "restart out (flag) = " << calflg_.restart_out_frg  << std::endl;
	std::cout << "// cout_CalFlag //" << std::endl;
}


void	paramCalFlag::
cout_CalFlag (const CalFlag	&calflg)
{
	std::cout << "restart            = " << calflg.restart << std::endl;
	std::cout << "cout, fout  (step) = " << calflg.cout_step << ", " << calflg.fout_step << std::endl;
	std::cout << "cout, fout  (flag) = " << calflg.cout_frg  << ", " << calflg.fout_frg  << std::endl;
	std::cout << "restart out (flag) = " << calflg.restart_out_frg  << std::endl;
}


// paramCalFlag //
