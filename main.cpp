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


void load_data(Matrix1D& A, Matrix1D& F, Matrix1D& X, int DIMENSION)
{
    srand(time(0));
//    cout << "Load data entry...";
    for (int i = 0; i < DIMENSION; i++) {
//        cout << "Alive entry Load_Data" << i << endl;
        for (int j = 0; j < DIMENSION; j++) {
//            cout << "I alive" << j << endl;
            if (i == j)
                A[i+j] = 100 * DIMENSION + rand() % 300 * DIMENSION;
            else
                A[i+j] = 1 + rand() % 10;
//            cout << A[i+j] << endl;
        }
        F[i] = 1 + rand() % 10;
        X[i] = 1;
    }
    cout << "Dataload finished!" << endl;
}


//void display_1d(Matrix1D& arg)
//{
//    for (size_t i = 0; i < DIMENSION; i++)
//        cout << arg[i] << endl;
//    cout << "=========================";
//}
//
//
//void display_2d(Matrix2D& arg)
//{
//    for (size_t i = 0; i < DIMENSION; i++) {
//        for (int g = 0; g < DIMENSION; g++)
//            cout << arg[i][g] << " ";
//        cout << endl;
//    }
//    cout << endl;
//}


void approx_init(Matrix1D& X)
{
    for (size_t i = 0; i < X.size(); i++)
        X[i] = 1;
}


/// N - размерность матрицы; A[N][N] - матрица коэффициентов, F[N] - столбец свободных членов,
/// X[N] - начальное приближение, также ответ записывается в X[N];

//void solve_worker(int iternum, Matrix1D& A, Matrix1D& X, Matrix1D& F, int startIndex, int endIndex, int N)
//{
//    int col_len = N;
//    Matrix1D TempX(endIndex-startIndex);
//    cout << "Entry worker, index: " << endIndex << endl;
//    for (int run = 0; run < iternum; run++)
//    {
//        for (int i = startIndex; i <= endIndex; i++)
//        {
//            TempX[i] = F[i];
//            for (int g = 0; g < col_len; g++)
//            {
//                if (i != g)
//                    TempX[i] -= A[i+g] * X[g];
//                cout << "worker from i,g: [" << i << "," << g << "]" << endl;
//            }
//            TempX[i] /= A[i+i];
//        }
//        cout << "Eror?" << endl;
//        // Here was 'norm' calculation - it's deprecated now.
//    }
//}


void solve_worker(int iternum, Matrix1D* A, Matrix1D* X, Matrix1D* F, int startIndex, int endIndex, int N, int rank)
{
    int col_len = N;
    Matrix1D TempX(endIndex-startIndex);
    cout << "Entry worker, index: " << endIndex << endl;
    for (int run = 0; run < iternum; run++)
    {
        for (int i = startIndex; i <= endIndex; i++)
        {
            TempX[i] = F->at(i);
            for (int g = 0; g < col_len; g++)
            {
                if (i != g)
                    TempX[i] -= A->at(i+g) * X->at(g);
//                cout << "worker from i,g: [" << i << "," << g << "]" << endl;
            }
            TempX[i] /= A->at(i+i);
        }
        cout << "Eror in proc" << rank << endl;
        // Here was 'norm' calculation - it's deprecated now.
    }
}





int main(int argc, char* argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    int iterations = 1; ////////////////////////
    int dimension  = 100;     ///////////////////////
    int startIndex = 0;
    int endIndex   = 0;
    int chunk      = dimension / (size-1);

    Matrix1D A(dimension * dimension);  //N*N
    Matrix1D F(dimension);  //N
    Matrix1D X(dimension);  //N

    if (dimension % (size-1) != 0) {
        MPI_Finalize();
        cout << "Processes_num should be divisor of dimension!";
        return 0;
    }

    if (rank == 0) {
        cout << "Main processor, total procs: " << size << endl;
//        init(dimension, A, F, X);
        load_data(A, F, X, dimension);

        for (int i=1; i < size; i++) {
            startIndex = (dimension / (size - 1)) * (i - 1);
            endIndex = ((dimension / (size - 1)) * i) - 1;
            cout << "Master process sends to process#" << i << " values [start, end]: " << startIndex << " - " << endIndex << endl;
            MPI_Send(&startIndex, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&endIndex, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
//            cout << "Send ok #" << i <<  endl;
        }
    }
    else if (rank != 0) {
        MPI_Recv(&startIndex, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&endIndex, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cout << "Process#" << rank << " received values: [" << startIndex << "," << endIndex << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    cout << "Starting bcast" << endl;
    MPI_Bcast(&A[0], A.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    cout << "A -- cast completed" << endl;
    MPI_Bcast(&F[0], F.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    cout << "F -- cast completed" << endl;
    MPI_Bcast(&X[0], X.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    cout << "X -- cast completed" << endl;

    if (rank != 0) {
        cout << "worker launched from rank:" << rank << endl;
                Matrix1D TempX(endIndex-startIndex);
                for (int run = 0; run < iterations; run++)
                {
                    for (int i = startIndex; i <= endIndex; i++)
                    {
                        TempX[i] = F[i];
                        for (int g = 0; g < dimension; g++)
                        {
                            if (i != g)
                                TempX[i] -= A[i+g] * X[g];
//                            cout << "worker from i,g: [" << i << "," << g << "]" << endl;
                        }
                        TempX[i] /= A[i+i];
                    }
                    cout << "Eror?" << endl;
                    // Here was 'norm' calculation - it's deprecated now.
                }

//        solve_worker(iterations, &A, &X, &F, startIndex, endIndex, dimension, rank);
    }
    //solve_worker(iterations, A, X, F, startIndex, endIndex);
    MPI_Barrier(MPI_COMM_WORLD);

    cout << "Completed " << endl;


    return 0;
}
