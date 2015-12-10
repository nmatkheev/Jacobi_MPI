#include <iostream>
#include <fstream>

#include <cmath>

#include <mpi.h>

#include <vector>
#include <array>
#include <algorithm>

const double EPS = 0.0000001;


using namespace std;
using Matrix1D = std::vector<double>;
using Matrix2D = std::vector<std::vector<double> >;


// Global declaration -------------------------------//
int DIMENSION;
Matrix2D A;
Matrix1D F;
Matrix1D X;
double time_simp;
double time_omp;

void load_data();
void solve_casual();
void solve_omp();
void init_data();
void display_1d(Matrix1D);
void display_2d(Matrix2D);

void init(int N) {
    srand(time(0));
    DIMENSION = N;

    A.resize(DIMENSION);
    F.resize(DIMENSION);
    X.resize(DIMENSION);

    for (size_t i = 0; i<N; i++) {
        A[i].resize(N);
    }

    time_omp = 0;
    time_simp = 0;

    cout << "Constructor is ok!" << endl;
}
// Global declaration end ----------------------------//


void load_data()
{
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
        //cin >> F[i];
        X[i] = 1;
    }
}


void display_1d(Matrix1D arg)
{
    for (size_t i = 0; i < DIMENSION; i++)
        cout << arg[i] << endl;
    cout << "=========================";
}


void display_2d(Matrix2D arg)
{
    for (size_t i = 0; i < DIMENSION; i++) {
        for (int g = 0; g < DIMENSION; g++)
            cout << arg[i][g] << " ";
        cout << endl;
    }
    cout << endl;
}


void init_data()
{
    for (size_t i = 0; i < DIMENSION; i++)
        X[i] = 1;
}


/// N - размерность матрицы; A[N][N] - матрица коэффициентов, F[N] - столбец свободных членов,
/// X[N] - начальное приближение, также ответ записывается в X[N];

void solve_worker(int iternum, Matrix2D* sliceA, Matrix1D* sliceX, Matrix1D* sliceF)
{
    int row_len = sliceA->size();
    int col_len = sliceA->at(0).size();
    int g;
    Matrix1D TempX(row_len);

    for (int i = 0; i<iternum; i++)
    {
        for (int i = 0; i < row_len; i++)
        {
            TempX[i] = sliceF->at(i);
            for (g = 0; g < col_len; g++)
            {
                if (i != g)
                    TempX[i] -= sliceA->at(i).at(g) * sliceX->at(g);
            }
            TempX[i] /= sliceA->at(i).at(i);
        }
        // Here was 'norm' calculation - it's deprecated now.
    }
}


int main(int argc, char* argv[])
{
    int n, rank, size;
    int dimension = 1000;
    double startwtime = 0.0; double endwtime;

    const int iterations = 10;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    if (dimension % (size-1) != 0) {
        MPI_Finalize();
        cout << "Processes_num should be divisor of dimension!";
        return 0;
    }

    if (rank == 0)
    {
        startwtime = MPI_Wtime();
        cout << "Main processor, total procs: " << size << "\n";

        cout << "Creating data-structures... " << endl;
        init(dimension);

        cout << "Loading data into structures... " << endl;
        load_data();
//        solve_casual();
        cout << "Init the starting approxumation vector..." << endl;
        init_data();
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {
        Matrix2D sliceA(dimension/size);
        Matrix1D sliceX(dimension/size);
        Matrix1D sliceF(dimension/size);

        for (int i=0; i<dimension/size; i++)
        {
            sliceA.push_back(A[i + ((dimension/size) * rank)]);
            sliceX.push_back(X[i + ((dimension/size) * rank)]);
            sliceF.push_back(X[i + ((dimension/size) * rank)]);
        }
        solve_worker(iterations, &sliceA, &sliceX, &sliceF);
    }


    cout << "Casual time: " << time_simp << endl;
    cout << "OpenMP time: " << time_omp << endl;


    return 0;
}
