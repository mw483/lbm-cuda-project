/**
 * @file option_parser-user-def.h
 * @brief (File brief)
 *
 * (File explanation)
 *
 * @author Takashi Shimokawabe
 * @date 2010/08/26 Created
 * @version 0.1.0
 *
 * $Id: option_parser-user-def.h,v 8c9c367349ca 2011/01/06 10:25:12 shimokawabe $ 
 */


/*
  Available type:
      int, float, double, std::string.
      You may also use other types. However, you can not use "char *".
  
  Available options:

  For Help.
      OPTION_PARSER_HELP   ( help    , "-h" , "--help")

  For Version information.
      OPTION_PARSER_VERSION( version , "-v" , "--version", "Version 10.0.2")

  Only print message. 
      OPTION_PARSER_MESSAGE( author  , "-a" , "--author" , "Author: Takashi Shimokawabe", "Show the author")

  For a flag (required no argument).
      OPTION_PARSER_OPT    ( flag   , "-f" , "--flag"    ,                                "flag option" )

  For a option required an argument.      
      OPTION_PARSER_OPTARG ( intopt , ""   , "--intopt"  , int        , 2000     , "N"  , "int option" )

  For a option required more than one argument.
  Use the keyword "G" for more than one argument.
      OPTION_PARSER_OPTARGS( intopts, ""   , "--intopts", int        , 2, G({10, 20}), G({"N1", "N2"}), "int options" )

  For a option required more than one specified keyword.
      OPTION_PARSER_OPTCHOICE( choice, "-c" , "--choice" , int        , 3, G({10, 11, 12}), 10, "C", "Choose from 10, 11, 12")

*/


// default
OPTION_PARSER_HELP   ( help    , "-h" , "--help")
OPTION_PARSER_VERSION( version , "-v" , "--version", "Version 10.0.2")
OPTION_PARSER_MESSAGE( author  , "-a" , "--author" , "Author: Takashi Shimokawabe", "Show the author")

OPTION_PARSER_OPT    ( flag    , "-f" , "--flag"   ,                                "flag option" )
OPTION_PARSER_OPTARG ( intopt  , "-i" , "--intopt" , int        , 3000     , "N"  , "Requires one argument" )
OPTION_PARSER_OPTARG ( dopt    , "-d" , "--dopt"   , double     , 100.0    , "N"  , "Requires one argument" )
OPTION_PARSER_OPTARG ( uopt    , "-u" , "--uopt"   , unsigned int, 20     , "N"  , "Requires one argument" )
OPTION_PARSER_OPTARG ( str     , "-s" , "--str"    , std::string, "string" , "STR", "Requires one argument" )
OPTION_PARSER_OPTARGS( intopts , "-i2", "--i2"      , int        , 2, G({10, 20}), G({"N1", "N2"}), "Requires two arguments" )

OPTION_PARSER_OPTCHOICE( choice, "-c" , "--choice" , int        , 3, G({10, 11, 12}), 10, "C", "Choose from 10, 11, 12")


// CUDA Code
// Calfrag
OPTION_PARSER_OPTARGS (
	cfout,
   	 "-CFout",
	"--CFout",
	int,
	2,
	G({100, 1000}),
	G({"cout step", "fout step"}),
	"Requires two arguments" )


OPTION_PARSER_OPTARGS (
	cfrfrg ,
	 "-CFRfrg",
	"--CFRfrg",
	int, 3,
	G({1, 1, 0}),
	G({"fout frg", "cout frg", "restart frg"}),
	"Requires three arguments" )


// fstep start
OPTION_PARSER_OPTARG (
	fstart,
	 "-fstart",
	"--fstart",
	int,
	0,
	"fstart",
	"Requires one argument" )


// Restart
OPTION_PARSER_OPTARG (
	restart,
	 "-restart",
	"--restart",
	int,
	0,
	"restart",
	"Requires one argument" )


// Time
OPTION_PARSER_OPTARG (
	time,
	 "-Time",
	"--Time",
	double,
	1.0,
	"Time",
	"Requires one argument" )


// Time param
OPTION_PARSER_OPTARG (
	time_coef,
	 "-time_coef",
	"--time_coef",
	double,
	1.0,
	"Time",
	"Requires one argument" )


// 代表格子点数 (N, AXIS(x:0, y:1, z:2))
OPTION_PARSER_OPTARGS ( 
	cnumgrid,
	 "-CNN",
   	"--CNN", 
	int, 2,
	G({128, 0}),
	G({"N" , "AXIS x:0, y:1, z:2" }),
	"Requires three arguments" )


// NX, NY, NZ
OPTION_PARSER_OPTARGS (
	numgrid,
	 "-NN",
	"--NN",
	int,
	3,
	G({128, 128, 128}),
	G({"NX", "NY", "NZ"}),
	"Requires three arguments" )


// MPI process
OPTION_PARSER_OPTARGS (
	nummpi,
	 "-NMPI", 
	"--NMPI",
	int,
	3,
	G({1, 1, 1}),
	G({"NCPU_X", "NCPU_Y", "NCPU_Z"}),
	"Requires three arguments" )


// MPI process
OPTION_PARSER_OPTARGS (
	ncpu_div,
	 "-ncpu_div", 
	"--ncpu_div",
	int,
	4,
	G({1, 1, 1, 1}),
	G({"NCPU DIV X", "NCPU DIV Y", "NCPU DIV Z", "NCPU PARTICLE"}),
	"Requires three arguments" )


// 計算領域の最小幅
OPTION_PARSER_OPTARGS (
	domain_min,
	 "-domain_min",
	"--domain_min",
	double , 3,
	G({0.0, 0.0, 0.0}),
	G({"xg_min" , "yg_min" , "zg_min" }),
	"Requires three arguments" )


// 計算領域幅の最小幅
OPTION_PARSER_OPTARGS (
	length,
	 "-length",
	"--length",
	double , 3,
	G({1.0, 1.0, 1.0}),
	G({"LX" , "LY" , "LZ" }),
	"Requires three arguments" )


// number of GPUs per node //
OPTION_PARSER_OPTARG (
	gpu_per_node,
	 "-gpu_per_node",
	"--gpu_per_node",
	int,
	3,
	"number of GPUs per node",
	"Requires one argument" )


// number of halo grid //
OPTION_PARSER_OPTARG (
	halo_grid,
	 "-halo_grid",
	"--halo_grid",
	int,
	1,
	"number of halo",
	"Requires one argument" )


// lattice boltzmann method //
OPTION_PARSER_OPTARGS (
	velocity_lbm,
	 "-velocity_lbm",
	"--velocity_lbm",
	double , 2,
	G({10.0, 0.04}),
	G({"velocity" , "cfl" }),
	"Requires two arguments" )


// particle *****
OPTION_PARSER_OPTARG ( particle    , "-particle" , "--particle"   , unsigned int, 10000     , "N"  , "Requires one argument" )


// generate particle
OPTION_PARSER_OPTARG ( generate_step    , "-generate_step" , "--generate_step"   , unsigned int, 50     , "N"  , "Requires one argument" )

// Particle Restart
OPTION_PARSER_OPTARG ( prestart , "-prestart"    , "--prestart" , int        , 0        , "restart"  , "Requires one argument" )

// Particle Restart
OPTION_PARSER_OPTARG ( pout , "-pout"    , "--pout" , int        , 100        , "pout"  , "Requires one argument" )

// particle source
OPTION_PARSER_OPTARGS( numsource , "-Nsource"         , "--Nsource"      , int        , 2, G({16, 16}), G({"NX", "NY"}), "Requires two arguments" )

// pstep start
OPTION_PARSER_OPTARG ( pstartstep , "-pstartstep"    , "--pstartstep" , int        , 0        , "pstartstep"  , "Requires one argument" )



// particle generate flag //
OPTION_PARSER_OPTARG (
	flag_particle_generate,
	 "-flag_particle_generate",
	"--fp_generate",
	int,
	0,
	"flag_particle_generate",
	"Requires one argument" )
