
#include "cumath.h"

#define XY_TO_GL_01( x , y , nx , ny ) ( ( x   +(y+1)* nx) * 3    * 3 + 1 )
#define XY_TO_GL_02( x , y , nx , ny ) ((((x-1)+(y+1)* nx) * 3+2) * 3 + 1 )
#define XY_TO_GL_03( x , y , nx , ny ) ((( x   + y   * nx) * 3+1) * 3 + 1 )

#define XY_TO_GL_11( x , y , nx , ny ) ((((x-1)+( y    + ny) * nx) * 3  ) * 3 + 1 )
#define XY_TO_GL_12( x , y , nx , ny ) ((( x   +( y    + ny) * nx) * 3+2) * 3 + 1 )
#define XY_TO_GL_13( x , y , nx , ny ) ((((x-1)+((y+1) + ny) * nx) * 3+1) * 3 + 1 )

extern "C" {

__global__ void cut_x(float *hmap , float *drill , int bx , float by , int bz , int nx , int ny )
{
	int idx = bx + threadIdx.x;
	int idy = bz + threadIdx.y;

	int idd = threadIdx.x + threadIdx.y * blockDim.x;
	float h = by + drill[idd];

	int id = XY_TO_GL_01( idx , idy , nx , ny );
	if( hmap[id] > h ) hmap[id] = h;
	id = XY_TO_GL_02( idx , idy , nx , ny );
	if( hmap[id] > h ) hmap[id] = h;
	id = XY_TO_GL_03( idx , idy , nx , ny );
	if( hmap[id] > h ) hmap[id] = h;
	id = XY_TO_GL_11( idx , idy , nx , ny );
	if( hmap[id] > h ) hmap[id] = h;
	id = XY_TO_GL_12( idx , idy , nx , ny );
	if( hmap[id] > h ) hmap[id] = h;
	id = XY_TO_GL_13( idx , idy , nx , ny );
	if( hmap[id] > h ) hmap[id] = h;
}

}

