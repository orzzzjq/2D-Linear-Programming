#include "helper.cuh"
 
__host__ __device__
double _fabs(double x)
{
	if (x > 0) return x;
	return -x;
}
 
__host__ __device__
int equals(double num1, double num2) {
    return _fabs(num1 - num2) < EPS ? TRUE : FALSE;
}
 
__host__ __device__
int strictly_larger(double num1, double num2) {
    return (num1 - EPS > num2) ? TRUE : FALSE;
}
 
__host__ __device__
int strictly_less(double num1, double num2) {
    return (num1 + EPS < num2) ? TRUE : FALSE;
}
 
__host__ __device__
double random_double_bounds(double left_bound, double right_bound) {
    float scale = rand() / (float) RAND_MAX;
    double modified_left = left_bound + EPS;
    double modified_right = right_bound - EPS;
    return modified_left + scale * (modified_right - modified_left);
}
 
__host__ __device__
double random_double() {
    return random_double_bounds(RANDOM_LEFT_BOUND, RANDOM_RIGHT_BOUND);
}