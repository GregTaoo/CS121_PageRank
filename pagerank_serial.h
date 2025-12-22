#pragma once

#include "graph.h"

double *pagerank_serial(const graph *g, double damping, double eps, int max_iter, double *pr);