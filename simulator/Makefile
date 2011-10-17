
run: csparks.so solid_kernel.cubin
	python puma.py 

csparks.so: sparks/csparks.pyx
	cd sparks ; $(MAKE) $(MFLAGS)

solid_kernel.cubin: solid_kernel.cu
	nvcc --cubin -arch sm_11 -I/usr/lib64/python2.7/site-packages/pycuda/../../../../include/pycuda solid_kernel.cu

