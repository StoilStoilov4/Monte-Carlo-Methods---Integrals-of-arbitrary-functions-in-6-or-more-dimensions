#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cstdlib>

double test_function_exp(const std::vector<double>& x) {
    double sum_sq = 0.0;
    for (double xi : x) {
        sum_sq += xi * xi;
    }
    return std::exp(-sum_sq);
}

double monte_carlo_integral(long long N, int d) {
    std::mt19937_64 gen(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::vector<double> x(d);
    double sum = 0.0;

    for (long long i = 0; i < N; i++) {
        for (int j = 0; j < d; j++) {
            x[j] = dist(gen);
        }
        sum += test_function_exp(x);
    }

    return sum / static_cast<double>(N); // volume([0,1]^d) = 1
}

int main(int argc, char* argv[]) {
    long long N = 1000000;
    int d = 6;

    if (argc > 1) N = std::atoll(argv[1]);
    if (argc > 2) d = std::atoi(argv[2]);

    if (N <= 0 || d <= 0) {
        std::cerr << "Error: N and d must be positive.\n";
        return 1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    double estimate = monte_carlo_integral(N, d);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << std::fixed << std::setprecision(12);
    std::cout << "Sequential Monte Carlo Integration (exp)\n";
    std::cout << "----------------------------------------\n";
    std::cout << "Samples (N): " << N << "\n";
    std::cout << "Dimension (d): " << d << "\n";
    std::cout << "Estimated value: " << estimate << "\n";
    std::cout << "Execution time(s): " << elapsed.count() << "\n";

    return 0;
}