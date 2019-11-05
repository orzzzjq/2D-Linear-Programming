#include "models.cuh"
 
__host__ __device__
line generate_line_from_abc(double param_a, double param_b, double param_c) {
    line new_line;
    new_line.param_a = param_a;
    new_line.param_b = param_b;
    new_line.param_c = param_c;
    new_line.slope_value = compute_slope(new_line);
    return new_line;
}
 
__host__ __device__
line generate_line_from_kb(double k, double b) {
    line new_line;
    new_line.param_a = - k;
    new_line.param_b = 1;
    new_line.param_c = b;
    new_line.slope_value = k;
    return new_line;
}
 
__host__ __device__
line generate_line_from_2points(point p1, point p2) {
    if (equals(p1.pos_x, p2.pos_x) && equals(p1.pos_y, p2.pos_y)) {
        return generate_line_from_abc(0,0,0);
    }
    line new_line;
    new_line.param_a = p2.pos_y - p1.pos_y;
    new_line.param_b = p1.pos_x - p2.pos_x;
    new_line.param_c = p1.pos_x * p2.pos_y - p2.pos_x * p1.pos_y;
    new_line.slope_value = compute_slope(new_line);
    return new_line;
}
 
__host__ __device__
point generate_point_from_xy(double pos_x, double pos_y) {
    point new_point;
    new_point.pos_x = pos_x;
    new_point.pos_y = pos_y;
    return new_point;
}
 
__host__ __device__
point generate_intersection_point(line line1, line line2) {
    if (is_parallel(line1, line2)) {
        return point(0, 0);
    }
    point new_point;
    new_point.pos_x = (line1.param_c * line2.param_b - line1.param_b * line2.param_c)
            / (line1.param_a * line2.param_b - line1.param_b * line2.param_a);
    new_point.pos_y = (line1.param_c * line2.param_a - line1.param_a * line2.param_c)
            / (line1.param_b * line2.param_a - line1.param_a * line2.param_b);
    return new_point;
}
 
__host__ __device__
double compute_slope(line line) {
    if (equals(line.param_b, 0)) {
        if (line.param_a > 0) {
            return - MAXFLOAT;
        }
        return MAXFLOAT;
    }
    return - line.param_a / line.param_b;
}

__host__ __device__
int is_parallel(line line1, line line2) {
    return equals(line1.param_a * line2.param_b, line1.param_b * line2.param_a);
}