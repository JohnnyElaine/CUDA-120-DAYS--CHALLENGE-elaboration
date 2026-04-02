#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>

// row major matrix
typedef struct {
    size_t height;
    size_t width;
    float *data;
} Matrix;

Matrix *matrixAdd(const Matrix *A, const Matrix *B);
Matrix *matrixMult(const Matrix *A, const Matrix *B);
void initMatrixValues(Matrix *M, float minValue, float maxValue);
bool compareMatricies(const Matrix *A, const Matrix *B);
bool verifyMatriciesForAdd(Matrix *A, Matrix *B);
bool verifyMatriciesForMult(Matrix *A, Matrix *B);
void printMatrix(Matrix *M);

#endif