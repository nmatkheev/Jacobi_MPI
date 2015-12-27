mpicxx -std=c++11 -o jacompi.o main.cpp
mpirun -np 5 ./jacompi.o