#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#include "graph.h"

double *pagerank_omp(const int num_threads, const graph *g, const graph *converse,
                     const double damping, const double eps, const int max_iter, double *pr) {
  omp_set_num_threads(num_threads);

  const int n = g->v;
  const double inv_n = 1.0 / n;

  double *pr_new        = aligned_alloc(64, sizeof(double) * n);
  double *pr_normalized = aligned_alloc(64, sizeof(double) * n);
  // double *pr_new        = malloc(sizeof(double) * n);
  // double *pr_normalized = malloc(sizeof(double) * n);
  int *out_w            = malloc(sizeof(int) * n);
  memset(out_w, 0, sizeof(int) * n);

#pragma omp parallel for schedule(static)
  for (int u = 0; u < n; ++u) {
    const int start = g->offset[u], end = g->offset[u + 1];
    int w = 0;
    for (int i = start; i < end; ++i) {
      w += g->m[i].w;
    }
    out_w[u] = w;
    pr[u] = inv_n;
  }

  for (int iter = 0; iter < max_iter; ++iter) {
    double diff = 0.0;
    double dangling_sum = 0.0;
    double base_score = 0.0;

#pragma omp parallel
    {
#pragma omp for schedule(static) reduction(+: dangling_sum)
      for (int i = 0; i < n; ++i) {
        if (out_w[i] != 0) {
          pr_normalized[i] = pr[i] * damping / (double) out_w[i];
        } else {
          dangling_sum += pr[i];
        }
      }

#pragma omp single
      base_score = (1.0 - damping + damping * dangling_sum) / n;

#pragma omp for schedule(dynamic, 64) reduction(+: diff)
      for (int u = 0; u < n; ++u) {
        double sum = base_score;
        const int start = converse->offset[u], end = converse->offset[u + 1];
        for (int i = start; i < end; ++i) {
          const int v = converse->m[i].v;
          const int w = converse->m[i].w;
          sum += w * pr_normalized[v];
        }
        pr_new[u] = sum;
        diff += fabs(sum - pr[u]);
      }
    }

    double *tmp = pr_new;
    pr_new = pr;
    pr = tmp;

    if (diff < eps) {
      break;
    }
  }

  free(pr_normalized);
  free(pr_new);
  free(out_w);
  return pr;
}
