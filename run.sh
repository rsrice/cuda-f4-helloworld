nvcc -arch=sm_80 -O3 --ptxas-options=-v --use_fast_math main.cu
cuobjdump a.out -sass -ptx > a.dump
