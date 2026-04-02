#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "math.h"
#include "matrix.h"

Matrix *matrixAdd(const Matrix *A, const Matrix *B)
{
    Matrix *C = (Matrix*) malloc(sizeof(Matrix));
    C->height = A->height;
    C->width = A->width;
    size_t N = C->height * C->width;
	C->data = (float*) malloc(N * sizeof(float));

    for (size_t row = 0; row < A->height; row++)
    {
        for (size_t col = 0; col < A->width; col++)
        {
            size_t i = row * C->width + col;
            C->data[i] = A->data[i] + B->data[i];
        }
    }

    return C;
}

Matrix *matrixMult(const Matrix *A, const Matrix *B)
{
    Matrix *C = (Matrix*) malloc(sizeof(Matrix));
    C->height = A->height;
    C->width = B->width;
    C->data = (float*) malloc(C->height * C->width * sizeof(float));

    for (size_t y = 0; y < C->height; y++)
    {
        for (size_t x = 0; x < C->width; x++)
        {
            float sum = 0.0f;
            size_t a_idx = 0;
            size_t b_idx = 0;

            // row * column
            for (size_t elem_idx = 0; elem_idx < A->width; elem_idx++)
            {
                a_idx = y * A->width + elem_idx; // A[y, elem_idx]
                b_idx = elem_idx * B->width + x; // B[elem_idx, x]
                sum += A->data[a_idx] * B->data[b_idx];
            }

            size_t c_idx = y * C->width + x;
            C->data[c_idx] = sum;
        }
    }

    return C;
}

void initMatrixValues(Matrix *M, float minValue, float maxValue)
{
    static bool isSeeded = false;
    if (!isSeeded)
    {
        srand((unsigned int)time(NULL));
        isSeeded = true;
    }

    if (minValue > maxValue)
    {
        float tmp = minValue;
        minValue = maxValue;
        maxValue = tmp;
    }

    for (size_t row = 0; row < M->height; row++)
    {
        for (size_t col = 0; col < M->width; col++)
        {
            size_t i = row * M->width + col;
            float t = (float) rand() / (float) RAND_MAX;
            M->data[i] = minValue + t * (maxValue - minValue);
        }
    }
}

bool compareMatricies(const Matrix *A, const Matrix *B)
{
    if (A->height != B->height || A->width != B->width)
    {
        printf("Matrix sizes do not match: A=%zux%zu, B=%zux%zu\n",
               A->height, A->width, B->height, B->width);
        return false;
    }

    const float absEps = 1e-4f;
    const float relEps = 1e-5f;
    bool isMatching = true;

    for (size_t row = 0; row < A->height; row++)
    {
        for (size_t col = 0; col < A->width; col++)
        {
            size_t i = row * A->width + col;
            float a = A->data[i];
            float b = B->data[i];
            float diff = fabsf(a - b);
            float scale = fmaxf(fabsf(a), fabsf(b));

            if (diff > absEps && diff > relEps * scale)
            {
                printf("Matrices at row=%zu, column=%zu do not match! A=%f, B=%f, diff=%f\n",
                       row, col, a, b, diff);
                isMatching = false;
            }
        }
    }

    return isMatching;
}

bool verifyMatriciesForAdd(Matrix *A, Matrix *B){
	if (A->height != B->height || A->width != B->width) 
    {
		return false;
    }

	return true;
}

bool verifyMatriciesForMult(Matrix *A, Matrix *B){
	if (A->width != B->height ) 
    {
        printf("Matrix sizes  fit  not match!\n");
		return false;
    }

	return true;
}

void printMatrix(Matrix *M)
{
	for (size_t row = 0; row < M->height; row++)
    {
        for (size_t col = 0; col < M->width; col++)
        {
            size_t i = row * M->width + col;
			printf("%f\t", M->data[i]);
        }
		printf("\n");
    }
}