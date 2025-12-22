#pragma once

#include "graph.h"

void pagerank_omp(const graph *g, double damping, double eps, int max_iter, double *pr);