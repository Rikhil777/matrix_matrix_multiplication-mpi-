#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <iomanip>

// Print matrix
void print_matrix(const std::vector<double>& mat, int rows, int cols, const std::string& name) {
    std::cout << name << " = " << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << std::setw(4) << mat[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Multiply rows of A with B and store in C
void multiply_rows(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C,
                  int M, int N, int K, int row_start, int row_end) {
    for (int i = row_start; i < row_end; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k) {
                sum += A[i*K + k] * B[k*N + j];
            }
            C[(i - row_start)*N + j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int M = 8, K = 8, N = 8; // Example matrix sizes

    std::vector<double> A, B, C;
    if (rank == 0) {
        // Initialize matrices with random values
        A.resize(M*K);
        B.resize(K*N);
        C.resize(M*N, 0.0);
        for (int i = 0; i < M*K; ++i) A[i] = rand() % 10;
        for (int i = 0; i < K*N; ++i) B[i] = rand() % 10;

        std::cout << "Initial Matrices:" << std::endl;
        print_matrix(A, M, K, "A");
        print_matrix(B, K, N, "B");
    }

    // Broadcast B to all processes
    if (rank == 0) {
        for (int p = 1; p < size; ++p) {
            MPI_Send(B.data(), K*N, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
        }
    } else {
        B.resize(K*N);
        MPI_Recv(B.data(), K*N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Divide rows of A among processes
    int rows_per_proc = M / size;
    int extra = M % size;
    int row_start = rank * rows_per_proc + std::min(rank, extra);
    int row_end = row_start + rows_per_proc + (rank < extra ? 1 : 0);
    int local_rows = row_end - row_start;

    std::vector<double> local_A(local_rows * K);
    std::vector<double> local_C(local_rows * N, 0.0);

    // Distribute rows of A
    if (rank == 0) {
        for (int p = 1; p < size; ++p) {
            int p_row_start = p * rows_per_proc + std::min(p, extra);
            int p_row_end = p_row_start + rows_per_proc + (p < extra ? 1 : 0);
            int p_rows = p_row_end - p_row_start;
            MPI_Send(A.data() + p_row_start*K, p_rows*K, MPI_DOUBLE, p, 1, MPI_COMM_WORLD);
        }
        // Copy own rows
        std::copy(A.begin() + row_start*K, A.begin() + row_end*K, local_A.begin());
    } else {
        MPI_Recv(local_A.data(), local_rows*K, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Start timer before multiplication
    double start = MPI_Wtime();

    // Compute local multiplication
    multiply_rows(local_A, B, local_C, local_rows, N, K, 0, local_rows);

    // Gather results manually
    if (rank == 0) {
        // Copy own result
        std::copy(local_C.begin(), local_C.end(), C.begin() + row_start*N);
        for (int p = 1; p < size; ++p) {
            int p_row_start = p * rows_per_proc + std::min(p, extra);
            int p_row_end = p_row_start + rows_per_proc + (p < extra ? 1 : 0);
            int p_rows = p_row_end - p_row_start;
            MPI_Recv(C.data() + p_row_start*N, p_rows*N, MPI_DOUBLE, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(local_C.data(), local_rows*N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }

    double end = MPI_Wtime();
    double elapsed = end - start;

    // Print result on rank 0
    if (rank == 0) {
        print_matrix(C, M, N, "C");
        std::cout << "Time taken for multiplication: " << elapsed << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
