rm conway.cu.o
rm conway 
/usr/local/cuda-10.1/bin/nvcc -ccbin g++ -I/usr/local/cuda/samples/common/inc  -m64     -o conway.cu.o -c conway.cu
/usr/local/cuda-10.1/bin/nvcc -ccbin g++ -m64 -o conway conway.cu.o 




