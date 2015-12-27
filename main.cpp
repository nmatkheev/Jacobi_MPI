#include <iostream>
#include <fstream>

#include <cmath>

#include "mpi/mpi.h"

#include <vector>
#include <array>
#include <algorithm>

const double EPS = 0.0000001;

using namespace std;
using Matrix1D = std::vector<double>;
using Matrix2D = std::vector<std::vector<double> >;


void load_data(Matrix2D& A, Matrix1D& F, Matrix1D& X, int DIMENSION)
{
    srand(time(0));

    for (int i = 0; i < DIMENSION; i++) {
        A[i].resize(DIMENSION);

        for (int j = 0; j < DIMENSION; j++) {
            if (i == j)
                A[i][j] = 100 * DIMENSION + rand() % 300 * DIMENSION;
            else
                A[i][j] = 1 + rand() % 10;
        }
        F[i] = 1 + rand() % 10;
        X[i] = 1;
    }
    cout << "Dataload finished!" << endl;
}

/// N - размерность матрицы; A[N][N] - матрица коэффициентов, F[N] - столбец свободных членов,
/// X[N] - начальное приближение, также ответ записывается в X[N];

void solve_worker(Matrix2D chunkA, Matrix1D chunkX, Matrix1D chunkF, int iternum, int N, int rank)
{
    cout << "Worker launched from process #" << rank << endl;
    cout << chunkA[0][0] << endl;
    cout << chunkF[0] << endl;
    cout << chunkX[0] << endl;
    Matrix1D TempX(chunkX.size());
    cout << "Worker launched from process #" << rank << endl;
    for (int run = 0; run < iternum; run++)
    {
        for (int i = 0; i < chunkA.size(); i++)
        {
            TempX[i] = chunkF[i];
            for (int g = 0; g < N; g++)
            {
                if (i != g)
                    TempX[i] -= chunkA[i][g] * chunkX[g];
            }
            TempX[i] /= chunkA[i][i];
        }
    }
    cout << "Worker #" << rank << " has completed normally" << endl;
}


int main(int argc, char* argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int iterations = 5;       ////////////////////////
    int dimension = 100;     ///////////////////////
    int startIndex = 0;
    int endIndex = 0;
    int chunk = dimension / (size - 1);


    if (dimension % (size - 1) != 0) {
        MPI_Finalize();
        cout << "Process number must be divisor of dimension!";
        return 0;
    }

    //for (int i = 1; i < size; i++)
    //{   // Set task for workers
        if (rank == 0)
        {
            Matrix2D A(dimension);  //N*N
            Matrix1D F(dimension);  //N
            Matrix1D X(dimension);  //N

            cout << "Main, total processors: " << size << endl;
            load_data(A, F, X, dimension);

            for (int i = 1; i < size; i++)
            {   // Set task for workers
                Matrix2D chunkA(chunk); // chunk*N
                for (int b = 0; b < chunk; b++) {
                    chunkA[b].resize(dimension);
                }
                Matrix1D chunkF(dimension);
                Matrix1D chunkX(dimension);

                startIndex = chunk * (i - 1);
                endIndex = chunk * i - 1;

                for (int j = startIndex; j <= endIndex; j++)
                {
//                    cout << "Startindex=" << startIndex << ", endIndex=" << endIndex << ", j=" << j << endl;
                    for (int x = 0; x < dimension; x++)
                    {
//                        cout << j-startIndex << endl;
                        chunkA[j - startIndex][x] = A[j][x];
                    }
                    chunkF[j - startIndex] = F[j];
                    chunkX[j - startIndex] = X[j];
//                    cout << "Slicing iteration #" << j << "has finished!" << endl;
                }
                cout << "Master sends to process #" << i << " array[start, end]: " << startIndex << " - " << endIndex << endl;
                MPI_Send(&chunkA.front(), chunkA.size(), MPI_FLOAT, i, 0, MPI_COMM_WORLD);
                cout << "Master sent to process # array A" << endl;
                MPI_Send(&chunkF.front(), chunkF.size(), MPI_FLOAT, i, 1, MPI_COMM_WORLD);
                cout << "Master sent to process # array F" << endl;
                MPI_Send(&chunkX.front(), chunkX.size(), MPI_FLOAT, i, 2, MPI_COMM_WORLD);
                cout << "Master sent to process # array X" << endl;
                cout << "Send for #" << i << " is OK!" << endl;
            }
        }
        else if (rank != 0)
        {
            cout << "Process " << rank << "reached this code" << endl;
            Matrix2D chunkA(chunk);
            for (int b = 0; b < chunk; b++) {
                chunkA[b].resize(dimension);
            }
            Matrix1D chunkF(dimension);
            Matrix1D chunkX(dimension);

            cout << "Process #"<< rank <<" -- Declaring buffers is ok!" << endl;

            MPI_Recv(&chunkA.front(), chunkA.size(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&chunkF.front(), chunkF.size(), MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&chunkX.front(), chunkX.size(), MPI_FLOAT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            cout << "Process #" << rank << " received values. " << endl;

            solve_worker(chunkA, chunkX, chunkF, iterations, dimension, rank);\
            cout << "Process #" << rank << " completed. " << endl;
        }
    //}
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}
