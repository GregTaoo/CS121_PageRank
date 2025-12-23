#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#include "graph.h"

double *pagerank_omp(const int num_threads, const graph *g, const graph *converse,
                  const double damping, const double eps, const int max_iter, double *pr) {
  omp_set_num_threads(num_threads);

  const int n = g->v;

  double *pr_new = aligned_alloc(64, sizeof(double) * n);
  int *out_w = malloc(sizeof(int) * n);
  memset(out_w, 0, sizeof(int) * n);
  double *pr_normalized = aligned_alloc(64, sizeof(double) * n);

#pragma omp parallel for schedule(static)
  for (int u = 0; u < n; u++) {
    pr[u] = 1.0 / n;
    int w = 0;
    for (int i = g->offset[u]; i < g->offset[u + 1]; i++) {
      w += g->m[i].w;
    }
    out_w[u] = w;
  }
  double dangling_sum = 0.0;

#pragma omp parallel for schedule(static) reduction(+: dangling_sum)
  for (int i = 0; i < n; i++) {
    if (out_w[i] == 0) {
      // dangling node
      dangling_sum += pr[i];
    }
  }

  for (int iter = 0; iter < max_iter; iter++) {
    const double dangling_contrib = damping * dangling_sum / n;
    double diff = 0.0;
    double new_dangling_sum = 0.0;

#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (int i = 0; i < n; i++) {
        if (out_w[i] > 0) {
          pr_normalized[i] = pr[i] * damping / (double) out_w[i];
        } else {
          pr_normalized[i] = 0.0;
        }
      }

#pragma omp for schedule(dynamic, 64)
      for (int u = 0; u < n; u++) {
        double sum = (1.0 - damping) / n + dangling_contrib;
        const int start = converse->offset[u], end = converse->offset[u + 1];
        for (int i = start; i < end; i++) {
          const int v = converse->m[i].v;
          const int w = converse->m[i].w;
          sum += w * pr_normalized[v];
        }
        pr_new[u] = sum;
      }

#pragma omp for schedule(static) reduction(+: diff, new_dangling_sum)
      for (int i = 0; i < n; i++) {
        diff += fabs(pr_new[i] - pr[i]);
        if (out_w[i] == 0) {
          new_dangling_sum += pr_new[i];
        }
      }
    }

    double *tmp = pr_new;
    pr_new = pr;
    pr = tmp;
    dangling_sum = new_dangling_sum;

    if (diff < eps) {
      break;
    }
  }

  free(pr_normalized);
  free(pr_new);
  free(out_w);
  return pr;
}
