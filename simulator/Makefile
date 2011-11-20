
run: csparks.so solid_kernel.cubin
	python c3sim.py 

csparks.so: sparks/csparks.pyx
	cd sparks ; $(MAKE) $(MFLAGS)

solid_kernel.cubin: solid_kernel.cu
	nvcc --cubin -arch sm_11 -I/usr/include/pycuda solid_kernel.cu

clean:
	rm solid_kernel.cubin

