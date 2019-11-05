#ifndef IO_H
#define IO_H

#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include "models.cuh"
#define DEFAULT_ANS_LEN 10000
#define N 100000

typedef struct inputs {
    int number;
    thrust::host_vector<line> lines;
    double obj_function_param_a;
    double obj_function_param_b;
} inputs;

typedef struct answer {
    double answer_b;
    line line1;
    line line2;
    point intersection_point;
	answer() {}
} answer;

inputs read_from_file(char * filename);
char * generate_ans_string(answer ans);

#endif //IO_H
