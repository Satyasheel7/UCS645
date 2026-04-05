#include <iostream>
#include <vector>
#include <chrono>
#include "function.h"

using namespace std;
using namespace chrono;

int main(int argc, char* argv[]) {
    // Problem size - adjust for experimentation or from command line
    int ROWS = 500;
    int COLS = 500;
    
    if (argc >= 3) {
        ROWS = atoi(argv[1]);
        COLS = atoi(argv[2]);
    }
    
    cout << "=====================================" << endl;
    cout << "Matrix-Vector Multiplication Lab" << endl;
    cout << "=====================================" << endl;
    cout << "Matrix size: " << ROWS << " x " << COLS << endl;
    cout << "Vector size: " << COLS << endl << endl;
    
    // Allocate memory
    vector<vector<double>> matrix(ROWS, vector<double>(COLS));
    vector<double> vec(COLS);
    vector<double> resultSeq(ROWS);
    vector<double> resultPar(ROWS);
    
    // Initialize data
    cout << "Initializing data..." << endl;
    initializeMatrix(matrix, ROWS, COLS);
    initializeVector(vec, COLS);
    
    // Sequential execution
    cout << "\n--- Sequential Execution ---" << endl;
    auto startSeq = high_resolution_clock::now();
    matrixVectorMultSeq(matrix, vec, resultSeq);
    auto endSeq = high_resolution_clock::now();
    auto durationSeq = duration_cast<milliseconds>(endSeq - startSeq);
    
    cout << "Time taken: " << durationSeq.count() << " ms" << endl;
    cout << "Result (first 5 elements): ";
    printVector(resultSeq, 5);
    
    // Parallel execution
    cout << "\n--- Parallel Execution (OpenMP) ---" << endl;
    auto startPar = high_resolution_clock::now();
    matrixVectorMultPar(matrix, vec, resultPar);
    auto endPar = high_resolution_clock::now();
    auto durationPar = duration_cast<milliseconds>(endPar - startPar);
    
    cout << "Time taken: " << durationPar.count() << " ms" << endl;
    cout << "Result (first 5 elements): ";
    printVector(resultPar, 5);
    
    // Performance analysis
    cout << "\n--- Performance Analysis ---" << endl;
    double speedup = (double)durationSeq.count() / durationPar.count();
    cout << "Speedup: " << speedup << "x" << endl;
    
    // Verify correctness
    bool correct = true;
    for (int i = 0; i < ROWS; i++) {
        if (abs(resultSeq[i] - resultPar[i]) > 1e-6) {
            correct = false;
            break;
        }
    }
    
    cout << "Results match: " << (correct ? "YES" : "NO") << endl;
    cout << "=====================================" << endl;
    //CORRELATION COMPUTATION
    cout << "\n\n=====================================" << endl;
    cout << "Correlation Computation" << endl;
    cout << "=====================================" << endl;

    // Flatten matrix (vector -> pointer style)
    float* data = new float[ROWS * COLS];

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            data[j + i * COLS] = matrix[i][j];
        }
    }

    float* corrSeq = new float[ROWS * ROWS];
    float* corrPar = new float[ROWS * ROWS];

    // Sequential correlation
    cout << "\n--- Correlation Sequential ---" << endl;
    auto startCorrSeq = high_resolution_clock::now();
    correlateSeq(ROWS, COLS, data, corrSeq);
    auto endCorrSeq = high_resolution_clock::now();

    auto durCorrSeq = duration_cast<milliseconds>(endCorrSeq - startCorrSeq);
    cout << "Time taken: " << durCorrSeq.count() << " ms" << endl;

    // Parallel correlation
    cout << "\n--- Correlation Parallel ---" << endl;
    auto startCorrPar = high_resolution_clock::now();
    correlatePar(ROWS, COLS, data, corrPar);
    auto endCorrPar = high_resolution_clock::now();

    auto durCorrPar = duration_cast<milliseconds>(endCorrPar - startCorrPar);
    cout << "Time taken: " << durCorrPar.count() << " ms" << endl;

    // Speedup
    double corrSpeedup = (double)durCorrSeq.count() / durCorrPar.count();
    cout << "\nCorrelation Speedup: " << corrSpeedup << "x" << endl;

    // Verify
    bool corrCorrect = true;
    for (int i = 0; i < ROWS * ROWS; i++) {
        if (abs(corrSeq[i] - corrPar[i]) > 1e-5) {
            corrCorrect = false;
            break;
        }
    }

    cout << "Correlation Results match: " 
         << (corrCorrect ? "YES" : "NO") << endl;

    delete[] data;
    delete[] corrSeq;
    delete[] corrPar;
    return 0;
}