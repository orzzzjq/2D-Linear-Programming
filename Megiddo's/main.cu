#include "helper.cuh"
#include "io.cuh"

using namespace std;

answer compute(inputs input);

int main() {
	// get the input data
	inputs input = read_from_file("./data");
	// get the answer
	long long t1 = clock();
	answer ans = compute(input);
	double compute_time = (double)(clock() - t1) / CLOCKS_PER_SEC;
	// 3display result and free memory
	char * ans_string = generate_ans_string(ans);
	printf("%s", ans_string);
	printf("Compute time: %fs\n", compute_time);

	getchar();
	return 0;
}

__host__ __device__
bool ccw(point a, point b) {
	return strictly_larger(a.pos_x * b.pos_y - a.pos_y * b.pos_x, 0);
}

struct rotation
{
	double a, b;
	__host__ __device__
	rotation(double a, double b)
		: a(a), b(b) {}

	__host__ __device__
	line operator()(line l)
	{
		line res;
		point obj_x(b, -a);
		point obj_y(a, b);
		point p1, p2, p_y(0, 1);
		if (l.slope_value < 1 && l.slope_value > -1) {
			line x1(1, 0, 10000);
			line x2(1, 0,-10000);
			p1 = generate_intersection_point(l, x1);
			p2 = generate_intersection_point(l, x2);

		}
		else {
			line y1(0, 1, 10000);
			line y2(0, 1,-10000);
			p1 = generate_intersection_point(l, y1);
			p2 = generate_intersection_point(l, y2);
		}

		point p3(p1.shadow(obj_x), p1.shadow(obj_y));
		point p4(p2.shadow(obj_x), p2.shadow(obj_y));

		res = generate_line_from_2points(p3, p4);
		if (strictly_larger(l.param_c, 0) ^ strictly_larger(res.param_c, 0)) {
			res.param_a *= -1;
			res.param_b *= -1;
			res.param_c *= -1;
		}
		res.upper = (res.param_b < 0);
		res.save = 1;
		return res;
	}
};

struct is_upper {
	__host__ __device__
	bool operator() (line l) { return l.upper; }
};

struct is_lower {
	__host__ __device__
	bool operator() (line l) { return !l.upper; }
};

struct tmpPoint
{
	point p;
	int lineNo1, lineNo2;
	__host__ __device__
	tmpPoint() {}
};

struct intersection
{
	thrust::device_vector<int>::iterator lineNo;
	thrust::device_vector<tmpPoint>::iterator tmp_points;
	thrust::device_vector<line>::iterator rotated_lines;
	
	__host__ __device__
	intersection(thrust::device_vector<int>::iterator lineNo,
		thrust::device_vector<tmpPoint>::iterator tmp_points,
		thrust::device_vector<line>::iterator rotated_lines)
		: lineNo(lineNo), tmp_points(tmp_points), rotated_lines(rotated_lines) {}
	
	__host__ __device__
	bool operator() (int idx)
	{
		if (idx % 2) return 0;
		int lineNo1 = *(lineNo + idx), lineNo2 = *(lineNo + idx + 1);
		tmpPoint new_tmpPoint;
		new_tmpPoint.p = generate_intersection_point(*(rotated_lines+lineNo1), *(rotated_lines + lineNo2));
		new_tmpPoint.lineNo1 = lineNo1;
		new_tmpPoint.lineNo2 = lineNo2;
		*(tmp_points + idx / 2) = new_tmpPoint;
		return 0;
	}
};

struct comparator {
	__host__ __device__
	bool operator() (tmpPoint a, tmpPoint b) {
		return a.p.pos_x < b.p.pos_x;
	}
};

struct line_cmp
{
	double x;
	__host__ __device__
	line_cmp(double x) : x(x) {}

	__host__ __device__
	bool operator() (line line1, line line2)
	{
		double y1 = (-line1.param_a * x + line1.param_c) / line1.param_b;
		double y2 = (-line2.param_a * x + line2.param_c) / line2.param_b;
		return y1 < y2;
	}
};

struct judge
{
	bool left;
	double median_x;
	thrust::device_vector<line>::iterator rotated_lines;

	__host__ __device__
	judge(bool left, double median_x,
		thrust::device_vector<line>::iterator rotated_lines)
		: left(left), median_x(median_x), rotated_lines(rotated_lines) {}

	__host__ __device__
	bool operator() (tmpPoint tmp_p)
	{
		point p = tmp_p.p;
		line line1 = *(rotated_lines + tmp_p.lineNo1);
		line line2 = *(rotated_lines + tmp_p.lineNo2);
		if (line1.upper ^ line2.upper); // one is I+ and the other is I-
		else if (left) { // on the left side
			if (p.pos_x - EPS < median_x);
			else {
				if (line1.upper) { // both are I-, remove the line with smaller slope
					if (strictly_less(line1.slope_value, line2.slope_value)) line1.save = 0;
					if (strictly_less(line2.slope_value, line1.slope_value)) line2.save = 0;
				}
				else { // both are I+, remove the line with larger slope
					if (strictly_larger(line1.slope_value, line2.slope_value)) line1.save = 0;
					if (strictly_larger(line2.slope_value, line1.slope_value)) line2.save = 0;
				}
			}
		}
		else { // on the right side
			if (p.pos_x + EPS > median_x);
			else {
				if (line1.upper) { // both are I-, remove the line with larger slope
					if (strictly_larger(line1.slope_value, line2.slope_value)) line1.save = 0;
					if (strictly_larger(line2.slope_value, line1.slope_value)) line2.save = 0;
				}
				else { // both are I+, remove the line with smaller slope
					if (strictly_less(line1.slope_value, line2.slope_value)) line1.save = 0;
					if (strictly_less(line2.slope_value, line1.slope_value)) line2.save = 0;
				}
			}
		}
		*(rotated_lines + tmp_p.lineNo1) = line1;
		*(rotated_lines + tmp_p.lineNo2) = line2;
		return 0;
	}
};

struct remove_or_not
{
	thrust::device_vector<line>::iterator rotated_lines;

	__host__ __device__
	remove_or_not(thrust::device_vector<line>::iterator rotated_lines)
		: rotated_lines(rotated_lines) {}

	__host__ __device__
	bool operator() (int idx)
	{
		line l = *(rotated_lines + idx);
		return l.save;
	}
};

struct saved
{
	__host__ __device__
	bool operator() (line l)
	{
		return l.save;
	}
};

#define lines input.lines

answer compute(inputs input) {
	answer ans;
	int num = input.number;

	// copy data to gpu
	thrust::device_vector <line> d_lines = lines;
	thrust::device_vector <line> rotated_lines(num);
	thrust::device_vector <line> upper_lines(num);
	thrust::device_vector <line> lower_lines(num);

	// rotation
	thrust::transform(d_lines.begin(), d_lines.end(), rotated_lines.begin(),
		rotation(input.obj_function_param_a, input.obj_function_param_b));
	
	// divide the lines into two parts
	thrust::partition(rotated_lines.begin(), rotated_lines.end(), is_upper());
	thrust::host_vector <line> h_lines(num);
	thrust::copy(rotated_lines.begin(), rotated_lines.begin() + num, h_lines.begin());
	thrust::copy_if(rotated_lines.begin(), rotated_lines.end(), upper_lines.begin(), is_upper());
	thrust::copy_if(rotated_lines.begin(), rotated_lines.end(), lower_lines.begin(), is_lower());
	int upper_num = thrust::count_if(rotated_lines.begin(), rotated_lines.end(), is_upper());
	int lower_num = num - upper_num;

	if (!lower_num) { //no answer
		ans.answer_b = MAXFLOAT;
		ans.intersection_point = point(0, 0);
		ans.line1 = generate_line_from_abc(0, 0, 0);
		ans.line2 = generate_line_from_abc(0, 0, 0);
		return ans;
	}
	
	thrust::device_vector <bool> useless(num);
	thrust::device_vector <int> lineNo(num);
	thrust::sequence(lineNo.begin(), lineNo.end());
	thrust::device_vector <int> Idx(num);
	thrust::sequence(Idx.begin(), Idx.end());
	thrust::device_vector <tmpPoint> tmp_points(num);
	// remove n/4 lines each time
	// until there are less than 10 lines
	while (num > 10) {
		// Partition the lines into pairs
		// and compute the intersection point of every pair
		int point_num = num / 2;
		thrust::transform(Idx.begin(), Idx.begin() + num, useless.begin(),
			intersection(lineNo.begin(), tmp_points.begin(), rotated_lines.begin()));

		// sort the points in the x-order
		// find the median point
		thrust::sort(tmp_points.begin(), tmp_points.begin() + point_num, comparator());
		tmpPoint median_p = tmp_points[point_num / 2];
		double median_x = median_p.p.pos_x;

		// find the highest I+ and the lowest I-
		int lower_idx = thrust::max_element(lower_lines.begin(),
			lower_lines.begin() + lower_num, line_cmp(median_x)) - lower_lines.begin();
		int upper_idx = thrust::min_element(upper_lines.begin(), 
			upper_lines.begin() + upper_num, line_cmp(median_x)) - upper_lines.begin();

		// determine which side of the test line gives the answer
		pair <double, double> H, U, L;
		line max_lower = lower_lines[lower_idx];
		line min_upper = upper_lines[upper_idx];
		L.first = (- max_lower.param_a * median_x + max_lower.param_c) / max_lower.param_b;
		L.second = max_lower.slope_value;
		if (upper_num == 0) U.first = MAXFLOAT, U.second = 0;
		else {
			U.first = (- min_upper.param_a * median_x + min_upper.param_c) / min_upper.param_b;
			U.second = min_upper.slope_value;
		}
		H.first = U.first - L.first;
		H.second = U.second - L.second;
		
		bool left, noAns = 0;
		if (equals(H.first, 0)) {                   // h(x)=0
			if (L.second < 0) left = 0;             // right
			else left = 1;                          // left   
		}
		else if (H.first < 0) {                     // h(x)<0
			if (equals(H.second, 0)) noAns = 1;     // no answer
			else if (H.second < 0) left = 1;        // left
			else left = 0;                          // right
		}
		else {                                      // h(x)>0
			if (L.second > 0) left = 1;             // left
			else left = 0;                          // right
		}

		if (noAns) { // no answer
			ans.answer_b = MAXFLOAT;
			ans.intersection_point = point(0, 0);
			ans.line1 = generate_line_from_abc(0, 0, 0);
			ans.line2 = generate_line_from_abc(0, 0, 0);
			return ans;
		}

		// judge every point and mark some useless lines
		thrust::transform(tmp_points.begin(), tmp_points.begin() + point_num, 
			useless.begin(), judge(left, median_x, rotated_lines.begin()));

		// remove the useless lines
		thrust::copy_if(Idx.begin(), Idx.end(), lineNo.begin(), remove_or_not(rotated_lines.begin()));
		num = thrust::count_if(rotated_lines.begin(), rotated_lines.end(), saved());
	}
	
	// copy the line number to cpu
	thrust::host_vector <int> h_lineNo(num);
	thrust::copy(lineNo.begin(), lineNo.begin() + num, h_lineNo.begin());

	// compute the answer
	double res_x, min_y = MAXFLOAT;
	int lineNo1 = 0, lineNo2 = 0;
	for (int i = 0; i < num; i++) {
		for (int j = i + 1; j < num; j++) {
			line l1 = h_lines[h_lineNo[i]];
			line l2 = h_lines[h_lineNo[j]];
			if (is_parallel(l1, l2)) continue;
			point p = generate_intersection_point(l1, l2);
			bool flag = 1;
			for (int k = 0; k < num; k++) {
				if (i == k || j == k) continue;
				line l3 = h_lines[h_lineNo[k]];
				double _y = (-l3.param_a * p.pos_x + l3.param_c) / l3.param_b;
				if (l3.upper && strictly_larger(p.pos_y, _y)) { flag = 0; break; }
				if ((!l3.upper) && strictly_less(p.pos_y, _y)) { flag = 0; break; }
			}
			if (flag) {
				if (p.pos_y < min_y) {
					res_x = p.pos_x;
					min_y = p.pos_y;
					lineNo1 = h_lineNo[i];
					lineNo2 = h_lineNo[j];
				}
			}
		}
	}

	printf("Rotated point: (%f, %f)\n", res_x, min_y);

	// rotate back
	line l1 = h_lines[lineNo1];
	line l2 = h_lines[lineNo2];

	point obj_x(input.obj_function_param_b, input.obj_function_param_a);
	point obj_y(-input.obj_function_param_a, input.obj_function_param_b);
	point p1, p2, p3, p4;
	if (l1.slope_value < 1 && l1.slope_value > -1) {
		line x1(1, 0, 10000);
		line x2(1, 0, -10000);
		p1 = generate_intersection_point(l1, x1);
		p2 = generate_intersection_point(l1, x2);

	}
	else {
		line y1(0, 1, 10000);
		line y2(0, 1, -10000);
		p1 = generate_intersection_point(l1, y1);
		p2 = generate_intersection_point(l1, y2);
	}
	p3.pos_x = p1.shadow(obj_x), p3.pos_y = p1.shadow(obj_y);
	p4.pos_x = p2.shadow(obj_x), p4.pos_y = p2.shadow(obj_y);
	l1 = generate_line_from_2points(p3, p4);
	if (l2.slope_value < 1 && l2.slope_value > -1) {
		line x1(1, 0, 10000);
		line x2(1, 0, -10000);
		p1 = generate_intersection_point(l2, x1);
		p2 = generate_intersection_point(l2, x2);

	}
	else {
		line y1(0, 1, 10000);
		line y2(0, 1, -10000);
		p1 = generate_intersection_point(l2, y1);
		p2 = generate_intersection_point(l2, y2);
	}
	p3.pos_x = p1.shadow(obj_x), p3.pos_y = p1.shadow(obj_y);
	p4.pos_x = p2.shadow(obj_x), p4.pos_y = p2.shadow(obj_y);
	l2 = generate_line_from_2points(p3, p4);
	
	ans.line1 = l1;
	ans.line2 = l2;
	ans.intersection_point = generate_intersection_point(l1, l2);
	ans.answer_b = ans.intersection_point.pos_x * input.obj_function_param_a
		+ ans.intersection_point.pos_y * input.obj_function_param_b;
	return ans;
}
