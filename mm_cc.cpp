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
void multiply_rows(const std::vector<double>& A, const std::vector<double>& B,
                   std::vector<double>& C, int local_rows, int N, int K) {
    for (int i = 0; i < local_rows; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k) {
                sum += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int M = 8, K = 8, N = 8; // Example sizes

    std::vector<double> A, B, C;
    if (rank == 0) {
        // Initialize matrices
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
    if (B.size() != K*N) B.resize(K*N);
    MPI_Bcast(B.data(), K*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Prepare to scatter rows of A
    int rows_per_proc = M / size;
    int extra = M % size;
    std::vector<int> sendcounts(size), displs(size);

    for (int i = 0; i < size; ++i) {
        int start = i * rows_per_proc + std::min(i, extra);
        int end = start + rows_per_proc + (i < extra ? 1 : 0);
        sendcounts[i] = (end - start) * K;
        displs[i] = start * K;
    }

    int local_rows = sendcounts[rank] / K;
    std::vector<double> local_A(sendcounts[rank]);
    std::vector<double> local_C(local_rows * N, 0.0);

    // Scatter rows of A
    MPI_Scatterv(A.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 local_A.data(), sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Start timing
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    // Compute local multiplication
    multiply_rows(local_A, B, local_C, local_rows, N, K);

    // Prepare to gather results
    std::vector<int> recvcounts(size), recvdispls(size);
    for (int i = 0; i < size; ++i) {
        int start = i * rows_per_proc + std::min(i, extra);
        int end = start + rows_per_proc + (i < extra ? 1 : 0);
        recvcounts[i] = (end - start) * N;
        recvdispls[i] = start * N;
    }

    if (rank == 0 && C.size() != M*N) C.resize(M*N);

    // Gather results into C
    MPI_Gatherv(local_C.data(), local_rows*N, MPI_DOUBLE,
                C.data(), recvcounts.data(), recvdispls.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // End timing
    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    double elapsed = end - start;

    // Print results (root only)
    if (rank == 0) {
        print_matrix(C, M, N, "C");
        std::cout << "Time taken for multiplication: "
                  << elapsed << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
