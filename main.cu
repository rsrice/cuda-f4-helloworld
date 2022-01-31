__global__ void f4_1_kernel(const float *__restrict__ u) {
  int i = threadIdx.x * 4;
  float4 *u4_ = (float4 *) (&u[i]);
  float4 u4 = u4_[0];
  float4 v4;
  v4.x = u4.x + 1;
  v4.y = u4.y + 1;
  v4.z = u4.z + 1;
  v4.w = u4.w + 1;
}

static void f4_1() {
  int size = 16;
  int mem_size = sizeof(float) * 16;

  float *u = (float *)malloc(mem_size);
  for (int i = 0; i < size; i++) {
    u[i] = i;
  }

  float *d_u;
  cudaMalloc((void **)&d_u, mem_size);
  cudaMemcpy(d_u, u, mem_size, cudaMemcpyHostToDevice);

  f4_1_kernel<<<1, size / 4>>>(d_u);

  cudaFree(d_u);
  free(u);
}

int main() {
    f4_1();
    // f4_2();

    return 0;
}
