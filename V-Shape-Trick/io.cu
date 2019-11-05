#include "io.cuh"
 
inputs read_from_file(char * filename) {
    inputs input;
    FILE * input_file = fopen(filename, "r");
    if (input_file == NULL) {
        printf("Cannot open the file.");
        exit(1);
    }
    fscanf(input_file, "%d", &input.number);
    for (int line_no = 0; line_no < input.number; line_no++) {
        double param_a, param_b, param_c;
        fscanf(input_file, "%lf %lf %lf", &param_a, &param_b, &param_c);
        line new_line = generate_line_from_abc(param_a, param_b, param_c);
        input.lines.push_back(new_line);
    }
    fscanf(input_file, "%lf %lf", &input.obj_function_param_a, &input.obj_function_param_b);
    return input;
}
 
char * generate_ans_string(answer ans) {
    char * ans_string = (char *) malloc(sizeof(char) * DEFAULT_ANS_LEN);
    sprintf(ans_string, "Answer is: %lf\n"
            "Line 1 is: %lfx + %lfy >= %lf\n"
            "Line 2 is: %lfx + %lfy >= %lf\n"
            "Intersection point: (%lf, %lf)\n",
           ans.answer_b,
           ans.line1.param_a, ans.line1.param_b, ans.line1.param_c,
    ans.line2.param_a, ans.line2.param_b, ans.line2.param_c,
    ans.intersection_point.pos_x, ans.intersection_point.pos_y);
    return ans_string;
}