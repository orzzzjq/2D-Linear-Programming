#ifndef HELPER_H
#define HELPER_H

#include <math.h>
#include <stdlib.h>
#include "heads.cuh"

#define TRUE    1
#define FALSE   0

#define EPS                 1e-7
#define RANDOM_LEFT_BOUND   (-10000)
#define RANDOM_RIGHT_BOUND  (+10000)

__host__ __device__ double _fabs(double x);
__host__ __device__ int equals(double num1, double num2);                               // num1 == num2
__host__ __device__ int strictly_larger(double num1, double num2);                      // num1 > num2
__host__ __device__ int strictly_less(double num1, double num2);                        // num1 < num2
__host__ __device__ double random_double_bounds(double left_bound, double right_bound); // (left_bound, right_bound)
__host__ __device__ double random_double();                                             //

#endif //HELPER_H
