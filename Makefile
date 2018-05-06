NVCC=nvcc
CC=g++

CUDA_INCLUDEPATH=/usr/local/cuda-8.0/include

NVCC_OPTS=-O3 -arch=sm_30 -Xcompiler -m64

GCC_OPTS=-O3 -m64 -std=c++0x

all: acp

acp: main.o ac.o ac_utils.o ac_serial.o ac_open_acc.o
	$(NVCC) $(NVCC_OPTS) -o acp main.o ac.o ac_utils.o ac_serial.o ac_open_acc.o

ac_open_acc.o: ac.h
	$(CC) -c $(GCC_OPTS) ac_open_acc.cpp -I $(CUDA_INCLUDEPATH)
ac_utils.o: ac_utils.h
	$(CC) -c $(GCC_OPTS) ac_utils.cpp -I $(CUDA_INCLUDEPATH)

ac_serial.o: ac.h
	$(CC) -c $(GCC_OPTS) ac_serial.cpp -I $(CUDA_INCLUDEPATH)

main.o: timer.h utils.h ac.h
	$(CC) -c $(GCC_OPTS) main.cpp -I $(CUDA_INCLUDEPATH)

ac.o: utils.h
	nvcc -c $(NVCC_OPTS) ac.cu

clean:
	rm -f *.o