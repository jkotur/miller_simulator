
__global__ void cut(float *hmap , float *drill , float *bounds )
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

      hmap[idx*3+1] -= drill[5];
}

