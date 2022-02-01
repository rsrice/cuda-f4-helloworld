#include <stdio.h>

__global__ void f_kernel(const float *__restrict__ u, float *__restrict__ v) {
  int i = threadIdx.x * 4;
  v[i+0] = u[i+0];
  v[i+1] = u[i+1];
  v[i+2] = u[i+2];
  v[i+3] = u[i+3];
}

__global__ void f4_kernel(const float *__restrict__ u, float *__restrict__ v) {
  int i = threadIdx.x * 4;
  float4 *u4_ = (float4 *) (&u[i]);
  float4 u4 = u4_[0];
  float4 v4;
  v4.x = u4.x;
  v4.y = u4.y;
  v4.z = u4.z;
  v4.w = u4.w;
  v[i+0] = v4.x;
  v[i+1] = v4.y;
  v[i+2] = v4.z;
  v[i+3] = v4.w;
}

__global__ void f4_p1_kernel(const float *__restrict__ u, float *__restrict__ v) {
  int i = threadIdx.x * 4;
  float4 *u4_ = (float4 *) (&u[i]);
  float4 u4 = u4_[0];
  float4 v4;
  v4.x = u4.x + 1;
  v4.y = u4.y + 1;
  v4.z = u4.z + 1;
  v4.w = u4.w + 1;
  v[i+0] = v4.x;
  v[i+1] = v4.y;
  v[i+2] = v4.z;
  v[i+3] = v4.w;
}

__global__ void f4_pv_kernel(const float *__restrict__ u, float *__restrict__ v) {
  int i = threadIdx.x * 4;

  float4 *u4_ = (float4 *) (&u[i]);
  float4 u4 = u4_[0];
  float4 *v4_ = (float4 *) (&v[i]);
  float4 v4 = v4_[0];

  v4.x = u4.x + v4.x;
  v4.y = u4.y + v4.y;
  v4.z = u4.z + v4.z;
  v4.w = u4.w + v4.w;

  lap4.x = u4.x + v4.x;
  lap4.y = u4.y + v4.y;
  lap4.z = u4.z + v4.z;
  lap4.w = u4.w + v4.w;

  v[i+0] = v4.x;
  v[i+1] = v4.y;
  v[i+2] = v4.z;
  v[i+3] = v4.w;
}

__global__ void f4_stencil_kernel(const float *__restrict__ u, float *__restrict__ v) {
  int i = 4;

  float4 *u4_ = (float4 *) (&u[i]);
  float4 u4_ijk = u4_[0];
  float4 u4_km1 = u4_[-1];
  float4 u4_kp1 = u4_[1];

  float4 lap;
  lap.x = u4_km1.w + u4_ijk.x + u4_ijk.y;
  lap.y = u4_ijk.x + u4_ijk.y + u4_ijk.z;
  lap.z = u4_ijk.y + u4_ijk.z + u4_ijk.w;
  lap.w = u4_ijk.z + u4_ijk.w + u4_kp1.x;

  v[i+0] = lap.x;
  v[i+1] = lap.y;
  v[i+2] = lap.z;
  v[i+3] = lap.w;
}

static void f4_p1() {
  int size = 16;
  int mem_size = sizeof(float) * 16;

  float *u = (float *)malloc(mem_size);
  for (int i = 0; i < size; i++) {
    u[i] = i;
  }

  float *d_u;
  cudaMalloc((void **)&d_u, mem_size);
  cudaMemcpy(d_u, u, mem_size, cudaMemcpyHostToDevice);

  float *d_v;
  cudaMalloc((void **)&d_v, mem_size);
  cudaMemcpy(d_v, u, mem_size, cudaMemcpyHostToDevice);

  f4_p1_kernel<<<1, size / 4>>>(d_u, d_v);

  cudaMemcpy(u, d_v, mem_size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < size; i++) {
    printf("%d:\t %f\n", i, u[i]);
  }

  cudaFree(d_v);
  cudaFree(d_u);
  free(u);
}

int main() {
    f4_p1();
    // f4_2();

    return 0;
}
