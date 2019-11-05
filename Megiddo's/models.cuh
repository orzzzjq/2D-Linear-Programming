#ifndef MODELS_H
#define MODELS_H

#include "helper.cuh"
#include "heads.cuh"

#define MAXFLOAT 1.99998*2e63

typedef struct line {
    // ax + by >= c
    double param_a;
    double param_b;
    double param_c;
	bool upper;
	bool save = 1;
    double slope_value;
	__host__ __device__
	line(double a=0, double b=0, double c=0)
		: param_a(a), param_b(b), param_c(c) {}
} line;

typedef struct point {
    // (x, y)
    double pos_x;
    double pos_y;
	__host__ __device__
	point(double _x=0, double _y=0) {
		pos_x = _x;
		pos_y = _y;
	}
	__host__ __device__
	double dist() {
		return sqrt((pos_x)*(pos_x) + (pos_y)*(pos_y));
	}
	__host__ __device__
	double dist(point b) {
		return sqrt((pos_x - b.pos_x)*(pos_x - b.pos_x) + (pos_y - b.pos_y)*(pos_y - b.pos_y));
	}
	__host__ __device__
	double dist(line l) {
		return fabs((l.param_a*pos_x + l.param_b*pos_y - l.param_c) /
			sqrt((l.param_a)*(l.param_a) + (l.param_b)*(l.param_b)));
	}
	__host__ __device__
	double shadow(point b) {
		return (b.pos_x * pos_x + b.pos_y * pos_y) /
			sqrt(b.pos_x * b.pos_x + b.pos_y * b.pos_y);
	}
} point;

// the functions below may not be all used

__host__ __device__ line generate_line_from_abc(double param_a, double param_b, double param_c); // ax + by = c
__host__ __device__ line generate_line_from_kb(double k, double b); // y = kx + b
__host__ __device__ line generate_line_from_2points(point p1, point p2); //

__host__ __device__ point generate_point_from_xy(double pos_x, double pos_y);
__host__ __device__ point generate_intersection_point(line line1, line line2);

__host__ __device__ double compute_slope(line line);
__host__ __device__ int is_parallel(line line1, line line2);

#endif //MODELS_H
