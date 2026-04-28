#ifndef DEFINEPARTICLEFLAG_H_
#define DEFINEPARTICLEFLAG_H_


// particle
//const int	PARTICLE_CAL  = 0;
//const int	PARTICLE_YP   = 1;
//const int	PARTICLE_YM   = 2;
//const int	PARTICLE_ZP   = 3;
//const int	PARTICLE_ZM   = 4;
//const int	PARTICLE_YPZP = 5;
//const int	PARTICLE_YMZP = 6;
//const int	PARTICLE_YPZM = 7;
//const int	PARTICLE_YMZM = 8;
//const int	PARTICLE_NA   = 255;

const int	NUM_MPI_PARTICLE = 26;

enum Particle_MPI_Tag { 
	tag_particle_cal,		// Calculation
	tag_particle_xp,		// MPI 1D
	tag_particle_xm,
	tag_particle_yp,
	tag_particle_ym,
	tag_particle_zp,
	tag_particle_zm,
	tag_particle_xpyp,		// MPI 2D
	tag_particle_xmyp,
	tag_particle_ypzp,
	tag_particle_ymzp,
	tag_particle_xpzp,
	tag_particle_xmzp,
	tag_particle_xpym,
	tag_particle_xmym,
	tag_particle_ypzm,
	tag_particle_ymzm,
	tag_particle_xpzm,
	tag_particle_xmzm,
	tag_particle_xpypzp,	// MPI 3D
	tag_particle_xmypzp,
	tag_particle_xpymzp,
	tag_particle_xpypzm,
	tag_particle_xmymzp,
	tag_particle_xpymzm,
	tag_particle_xmypzm,
	tag_particle_xmymzm,
	tag_particle_na
};


const Particle_MPI_Tag	PARTICLE_CAL    = tag_particle_cal;
const Particle_MPI_Tag	PARTICLE_XP     = tag_particle_xp;
const Particle_MPI_Tag	PARTICLE_XM     = tag_particle_xm;
const Particle_MPI_Tag	PARTICLE_YP     = tag_particle_yp;
const Particle_MPI_Tag	PARTICLE_YM     = tag_particle_ym;
const Particle_MPI_Tag	PARTICLE_ZP     = tag_particle_zp;
const Particle_MPI_Tag	PARTICLE_ZM     = tag_particle_zm;
const Particle_MPI_Tag	PARTICLE_XPYP   = tag_particle_xpyp;
const Particle_MPI_Tag	PARTICLE_XMYP   = tag_particle_xmyp;
const Particle_MPI_Tag	PARTICLE_YPZP   = tag_particle_ypzp;
const Particle_MPI_Tag	PARTICLE_YMZP   = tag_particle_ymzp;
const Particle_MPI_Tag	PARTICLE_XPZP   = tag_particle_xpzp;
const Particle_MPI_Tag	PARTICLE_XMZP   = tag_particle_xmzp;
const Particle_MPI_Tag	PARTICLE_XPYM   = tag_particle_xpym;
const Particle_MPI_Tag	PARTICLE_XMYM   = tag_particle_xmym;
const Particle_MPI_Tag	PARTICLE_YPZM   = tag_particle_ypzm;
const Particle_MPI_Tag	PARTICLE_YMZM   = tag_particle_ymzm;
const Particle_MPI_Tag	PARTICLE_XPZM   = tag_particle_xpzm;
const Particle_MPI_Tag	PARTICLE_XMZM   = tag_particle_xmzm;
const Particle_MPI_Tag	PARTICLE_XPYPZP = tag_particle_xpypzp;
const Particle_MPI_Tag	PARTICLE_XMYPZP = tag_particle_xmypzp;
const Particle_MPI_Tag	PARTICLE_XPYMZP = tag_particle_xpymzp;
const Particle_MPI_Tag	PARTICLE_XPYPZM = tag_particle_xpypzm;
const Particle_MPI_Tag	PARTICLE_XMYMZP = tag_particle_xmymzp;
const Particle_MPI_Tag	PARTICLE_XPYMZM = tag_particle_xpymzm;
const Particle_MPI_Tag	PARTICLE_XMYPZM = tag_particle_xmypzm;
const Particle_MPI_Tag	PARTICLE_XMYMZM = tag_particle_xmymzm;
const Particle_MPI_Tag	PARTICLE_NA     = tag_particle_na;


#endif
