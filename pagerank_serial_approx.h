#pragma once

#include "graph.h"

double *pagerank_serial_approx(const graph *g, const graph *converse, double damping, double eps, int max_iter, double *pr);