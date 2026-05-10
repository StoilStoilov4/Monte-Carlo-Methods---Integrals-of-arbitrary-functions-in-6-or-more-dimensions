#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <cstdlib>

double test_function_exp(const std::vector<double>& x) {
    double sum_sq = 0.0;
    for (double xi : x) {
        sum_sq += xi * xi;
    }
    return std::exp(-sum_sq);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long N = 1000000;
    int d = 6;

    if (argc > 1) N = std::atoll(argv[1]);
    if (argc > 2) d = std::atoi(argv[2]);

    if (N <= 0 || d <= 0) {
        if (rank == 0) {
            std::cerr << "Error: N and d must be positive.\n";
        }
        MPI_Finalize();
        return 1;
    }

    long long local_N = N / size;
    long long remainder = N % size;

    if (rank < remainder) {
        local_N++;
    }

    std::mt19937_64 gen(42 + rank * 1000);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::vector<double> x(d);
    double local_sum = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    for (long long i = 0; i < local_N; i++) {
        for (int j = 0; j < d; j++) {
            x[j] = dist(gen);
        }
        local_sum += test_function_exp(x);
    }

    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();
    double local_elapsed = end_time - start_time;

    double max_elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double estimate = global_sum / static_cast<double>(N);

        std::cout << std::fixed << std::setprecision(12);
        std::cout << "MPI Monte Carlo Integration (exp)\n";
        std::cout << "---------------------------------\n";
        std::cout << "Processes used: " << size << "\n";
        std::cout << "Samples (N): " << N << "\n";
        std::cout << "Dimension (d): " << d << "\n";
        std::cout << "Estimated value: " << estimate << "\n";
        std::cout << "Execution time(s): " << max_elapsed << "\n";
    }

    MPI_Finalize();
    return 0;
}