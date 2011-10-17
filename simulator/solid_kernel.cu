
#include "cumath.h"

#define XY_TO_GL_01( x , y , nx , ny ) ( ( x   +(y+1)* nx) * 3    )
#define XY_TO_GL_02( x , y , nx , ny ) ((((x-1)+(y+1)* nx) * 3+2) )
#define XY_TO_GL_03( x , y , nx , ny ) ((( x   + y   * nx) * 3+1) )

#define XY_TO_GL_11( x , y , nx , ny ) ((((x-1)+( y    + ny) * nx) * 3  ) )
#define XY_TO_GL_12( x , y , nx , ny ) ((( x   +( y    + ny) * nx) * 3+2) )
#define XY_TO_GL_13( x , y , nx , ny ) ((((x-1)+((y+1) + ny) * nx) * 3+1) )

extern "C" {

__global__ void cut_x(float3 *hmap , float3 *nmap , float *drill , int bx , float by , int bz , int nx , int ny )
{
	int idx = bx + threadIdx.x;
	int idy = bz + threadIdx.y;

	int idd = threadIdx.x + threadIdx.y * blockDim.x;
	float h = by + drill[idd];

	float3 n;
	int id;

	id = XY_TO_GL_01( idx , idy , nx , ny );
	if( hmap[id].y > h ) {
		hmap[id].y = h;
		n = cross( hmap[id] - hmap[id+1] , hmap[id] - hmap[id+2] );
		nmap[id] = nmap[id+1] = nmap[id+2] = normalize(-n);
	}

	id = XY_TO_GL_02( idx , idy , nx , ny );
	if( hmap[id].y > h ) {
		hmap[id].y = h;
		n = cross( hmap[id-2] - hmap[id-1] , hmap[id-2] - hmap[id] );
		nmap[id-2] = nmap[id-1] = nmap[id] = normalize(-n);
	}

	id = XY_TO_GL_03( idx , idy , nx , ny );
	if( hmap[id].y > h ) {
		hmap[id].y = h;
		n = cross( hmap[id-1] - hmap[id] , hmap[id-1] - hmap[id+1] );
		nmap[id-1] = nmap[id] = nmap[id+1] = normalize(-n);
	}

	id = XY_TO_GL_11( idx , idy , nx , ny );
	if( hmap[id].y > h ) {
		hmap[id].y = h;
		n = cross( hmap[id] - hmap[id+1] , hmap[id] - hmap[id+2] );
		nmap[id] = nmap[id+1] = nmap[id+2] = normalize(-n);
	}

	id = XY_TO_GL_12( idx , idy , nx , ny );
	if( hmap[id].y > h ) {
		hmap[id].y = h;
		n = cross( hmap[id-2] - hmap[id-1] , hmap[id-2] - hmap[id] );
		nmap[id-2] = nmap[id-1] = nmap[id] = normalize(-n);
	}

	id = XY_TO_GL_13( idx , idy , nx , ny );
	if( hmap[id].y > h ) {
		hmap[id].y = h;
		n = cross( hmap[id-1] - hmap[id] , hmap[id-1] - hmap[id+1] );
		nmap[id-1] = nmap[id] = nmap[id+1] = normalize(-n);
	}
}

}

