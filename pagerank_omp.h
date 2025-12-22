#pragma once

#include "graph.h"

double *pagerank_omp(int num_threads, const graph *g, const graph *converse, double damping, double eps, int max_iter, double *pr);