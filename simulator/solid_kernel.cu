
#include "cumath.h"

#define XY_TO_GL_01( x , y , px , py ) ( ( x   +(y+1)* px) * 3    )
#define XY_TO_GL_02( x , y , px , py ) ((((x-1)+(y+1)* px) * 3+2) )
#define XY_TO_GL_03( x , y , px , py ) ((( x   + y   * px) * 3+1) )

#define XY_TO_GL_11( x , y , px , py ) ((((x-1)+( y    + py) * px) * 3  ) )
#define XY_TO_GL_12( x , y , px , py ) ((( x   +( y    + py) * px) * 3+2) )
#define XY_TO_GL_13( x , y , px , py ) ((((x-1)+((y+1) + py) * px) * 3+1) )

extern "C" {

__global__ void cut_x(float3 *hmap , float3 *nmap , float *drill , int bx , float by , int bz , int px , int py , int nx , int ny )
{
	unsigned int itx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int ity = threadIdx.y + blockIdx.y * blockDim.y;

	if( itx >= nx || ity >= ny ) return;

	int idx = bx + itx;
	int idy = bz + ity;

	if( idx >= px || idy >= py ) return;

	int idd = itx + ity * nx;
	float h = by + drill[idd];

	float3 n;
	int id;

	id = XY_TO_GL_01( idx , idy , px , py );
	if( hmap[id].y > h ) {
		hmap[id].y = h;
		n = cross( hmap[id] - hmap[id+1] , hmap[id] - hmap[id+2] );
		nmap[id] = nmap[id+1] = nmap[id+2] = normalize(-n);
	}

	id = XY_TO_GL_02( idx , idy , px , py );
	if( hmap[id].y > h ) {
		hmap[id].y = h;
		n = cross( hmap[id-2] - hmap[id-1] , hmap[id-2] - hmap[id] );
		nmap[id-2] = nmap[id-1] = nmap[id] = normalize(-n);
	}

	id = XY_TO_GL_03( idx , idy , px , py );
	if( hmap[id].y > h ) {
		hmap[id].y = h;
		n = cross( hmap[id-1] - hmap[id] , hmap[id-1] - hmap[id+1] );
		nmap[id-1] = nmap[id] = nmap[id+1] = normalize(-n);
	}

	id = XY_TO_GL_11( idx , idy , px , py );
	if( hmap[id].y > h ) {
		hmap[id].y = h;
		n = cross( hmap[id] - hmap[id+1] , hmap[id] - hmap[id+2] );
		nmap[id] = nmap[id+1] = nmap[id+2] = normalize(-n);
	}

	id = XY_TO_GL_12( idx , idy , px , py );
	if( hmap[id].y > h ) {
		hmap[id].y = h;
		n = cross( hmap[id-2] - hmap[id-1] , hmap[id-2] - hmap[id] );
		nmap[id-2] = nmap[id-1] = nmap[id] = normalize(-n);
	}

	id = XY_TO_GL_13( idx , idy , px , py );
	if( hmap[id].y > h ) {
		hmap[id].y = h;
		n = cross( hmap[id-1] - hmap[id] , hmap[id-1] - hmap[id+1] );
		nmap[id-1] = nmap[id] = nmap[id+1] = normalize(-n);
	}
}

}

