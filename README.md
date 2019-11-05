This is a course project for Computational Geometry & GPGPU in NUS SoC 2018 Summer Workshop.

**Linear Programming (LP)** is a technique for the optimization of a linear objective function, subject to linear inequality constraints. Its feasible region is a convex polytope, which is a set defined as the intersection of finitely many half-spaces. Each linear programming algorithm finds a point in the polyhedron where this function has the smallest (or largest) value if such a point exists.

We used CUDA\THRUST toolkit to solve the 2D Linear Programming problem on the GPU. This repo includes a parallel implementation of [**Migiddo's Algorithm**](https://sarielhp.org/teach/2004/b/webpage/lec/27_lp_2d.pdf) and an advanced version with some tricks called **V-Shape** that runs faster than the original one.

Thank my teammates Xintong and Xinyi.
