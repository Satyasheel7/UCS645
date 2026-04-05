#include "function.h"
#include <iostream>
#include <random>
#include <omp.h>
#include <cmath>
using namespace std;

void initializeMatrix(vector<vector<double>>& matrix, int rows, int cols) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 10.0);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = dis(gen);
        }
    }
}

void initializeVector(vector<double>& vec, int size) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 10.0);
    
    for (int i = 0; i < size; i++) {
        vec[i] = dis(gen);
    }
}

void matrixVectorMultSeq(const vector<vector<double>>& matrix, 
                         const vector<double>& vec, 
                         vector<double>& result) {
    int rows = matrix.size();
    int cols = vec.size();
    
    for (int i = 0; i < rows; i++) {
        result[i] = 0.0;
        for (int j = 0; j < cols; j++) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
}

void matrixVectorMultPar(const vector<vector<double>>& matrix, 
                         const vector<double>& vec, 
                         vector<double>& result) {
    int rows = matrix.size();
    int cols = vec.size();
    
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        result[i] = 0.0;
        for (int j = 0; j < cols; j++) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
}

void printVector(const vector<double>& vec, int limit) {
    cout << "[";
    int size = vec.size();
    int printSize = (size < limit) ? size : limit;
    
    for (int i = 0; i < printSize; i++) {
        cout << vec[i];
        if (i < printSize - 1) cout << ", ";
    }
    
    if (size > limit) {
        cout << " ... (" << size << " elements total)";
    }
    cout << "]" << endl;
}

void correlateSeq(int ny, int nx, const float* data, float* result) {
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j <= i; j++) {

            double sum_i = 0, sum_j = 0;

            for (int k = 0; k < nx; k++) {
                sum_i += data[k + i*nx];
                sum_j += data[k + j*nx];
            }

            double mean_i = sum_i / nx;
            double mean_j = sum_j / nx;

            double num = 0, denom_i = 0, denom_j = 0;

            for (int k = 0; k < nx; k++) {
                double xi = data[k + i*nx] - mean_i;
                double xj = data[k + j*nx] - mean_j;

                num += xi * xj;
                denom_i += xi * xi;
                denom_j += xj * xj;
            }

            result[i + j*ny] = num / sqrt(denom_i * denom_j);
        }
    }
}

#include <omp.h>

void correlatePar(int ny, int nx, const float* data, float* result) {

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j <= i; j++) {

            double sum_i = 0, sum_j = 0;

            for (int k = 0; k < nx; k++) {
                sum_i += data[k + i*nx];
                sum_j += data[k + j*nx];
            }

            double mean_i = sum_i / nx;
            double mean_j = sum_j / nx;

            double num = 0, denom_i = 0, denom_j = 0;

            for (int k = 0; k < nx; k++) {
                double xi = data[k + i*nx] - mean_i;
                double xj = data[k + j*nx] - mean_j;

                num += xi * xj;
                denom_i += xi * xi;
                denom_j += xj * xj;
            }

            result[i + j*ny] = num / sqrt(denom_i * denom_j);
        }
    }
}