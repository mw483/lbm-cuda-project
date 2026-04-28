#ifndef FILELIB_H_
#define FILELIB_H_

#include <iostream>
#include <cstdlib>
#include <stdint.h>
#include <fstream>
#include "definePrecision.h"


namespace
fileLib {


// template
// ファイル出力 (サイズ + binary)
template<typename T>
void
write_file (
	T		*value,
	char	fname[],
	int		nn)
{
	FILE	*fp = fopen(fname, "w");

	const uint32_t	size   = nn*sizeof(T);
	fwrite(&size, sizeof(uint32_t), 1,  fp);
	fwrite(value, sizeof(T),        nn, fp);

	fclose(fp);
}


// ファイル入出力
template<typename T>
void
fread_fwrite (
	FILE	*fp_write, 
	FILE	*fp_read,
	int		nn)
{
	T	*tmp = new T[nn];
	fread (tmp, sizeof(T), nn, fp_read );
	fwrite(tmp, sizeof(T), nn, fp_write);
	delete [] tmp;
}


// ファイル入出力
template<typename T>
void
fread_fwrite_data (
	FILE	*fp_write, 
	FILE	*fp_read,
	int		nn)
{
	uint32_t	size;
	fread (&size, sizeof(uint32_t), 1, fp_read );
	fwrite(&size, sizeof(uint32_t), 1, fp_write);

	fread_fwrite <T> (fp_write, fp_read, nn);
}


} // namespace //


#endif
