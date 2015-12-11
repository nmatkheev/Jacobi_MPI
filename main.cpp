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



void init(int N, Matrix2D& A, Matrix1D& F, Matrix1D& X)
{
    A.resize(N);
    F.resize(N);
    X.resize(N);

    for (size_t i = 0; i<N; i++) {
        A[i].resize(N);
    }
    cout << "Constructor is ok!" << endl;
}


void load_data(Matrix2D& A, Matrix1D& F, Matrix1D& X, int DIMENSION)
{
    srand(time(0));
    cout << "Alive";
    for (int i = 0; i < DIMENSION; i++) {
//        cout << "Alive entry Load_Data" << i << endl;
        for (int j = 0; j < DIMENSION; j++) {
//            cout << "I alive" << j << endl;
            if (i == j)
                A[i][j] = 100 * DIMENSION + rand() % 300 * DIMENSION;
            else
                A[i][j] = 1 + rand() % 100;
        }
        F[i] = 1 + rand() % 10;
        X[i] = 1;
    }
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

void solve_worker(int iternum, Matrix2D& A, Matrix1D& X, Matrix1D& F, int startIndex, int endIndex)
{
    int col_len = A.at(0).size();
    Matrix1D TempX(endIndex-startIndex);

    for (int run = 0; run < iternum; run++)
    {
        for (int i = startIndex; i <= endIndex; i++)
        {
            TempX[i] = F.at(i);
            for (int g = 0; g < col_len; g++)
            {
                if (i != g)
                    TempX[i] -= A.at(i).at(g) * X.at(g);
            }
            TempX[i] /= A.at(i).at(i);
        }
        // Here was 'norm' calculation - it's deprecated now.
    }
}


int main(int argc, char* argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    int dimension = 10000;
//    double startwtime = 0.0;  double endwtime;
    int startIndex = 0; int endIndex = 0;

    Matrix2D A;
    Matrix1D F; Matrix1D X;

    const int iterations = 10; ////////////////////////


    if (dimension % (size-1) != 0)
    {
        MPI_Finalize();
        cout << "Processes_num should be divisor of dimension!";
        return 0;
    }

//    startwtime = MPI_Wtime();

    if (rank == 0)
    {
        cout << "Main processor, total procs: " << size << endl;

        init(dimension, A, F, X);
        load_data(A, F, X, dimension);

        MPI_Bcast(&A.front(), dimension, MPI_INT, 0, MPI_COMM_WORLD);
        cout << "A -- cast completed" << endl;
        MPI_Bcast(&F.front(), dimension, MPI_INT, 0, MPI_COMM_WORLD);
        cout << "F -- cast completed" << endl;
        MPI_Bcast(&X.front(), dimension, MPI_INT, 0, MPI_COMM_WORLD);
        cout << "X -- cast completed" << endl;
    }


    if (rank != 0)
    {
        solve_worker(iterations, A, X, F, startIndex, endIndex);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    cout << "Completed " << endl;


    return 0;
}
