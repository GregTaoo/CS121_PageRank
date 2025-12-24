#include <math.h>
#include <stdlib.h>

#include "graph.h"

double *pagerank_serial_approx(const graph *g, const graph *converse, const double damping,
                               const double eps, const int max_iter, double *pr) {
  const int n = g->v;
  const double inv_n = 1.0 / n;

  double *pr_new        = malloc(sizeof(double) * n);
  double *pr_normalized = malloc(sizeof(double) * n);
  int *out_w            = malloc(sizeof(int) * n);

  int *frontier      = malloc(sizeof(int) * n);
  int *next_frontier = malloc(sizeof(int) * n);
  int frontier_size  = n;

  for (int i = 0; i < n; i++) {
    pr[i] = inv_n;
    frontier[i] = i;
    int w = 0;
    for (int j = g->offset[i]; j < g->offset[i + 1]; j++) {
      w += g->m[j].w;
    }
    out_w[i] = w;
  }

  for (int iter = 0; iter < max_iter; iter++) {
    double dangling_sum = 0.0;
    for (int i = 0; i < frontier_size; i++) {
      const int u = frontier[i];
      if (out_w[u] != 0) {
        pr_normalized[u] = pr[u] * damping / (double) out_w[u];
      } else {
        dangling_sum += pr[u];
      }
    }

    const double base_score = (1.0 - damping + damping * dangling_sum) / n;

    int next_frontier_size = 0;
    for (int i = 0; i < frontier_size; i++) {
      const int u = frontier[i];
      double sum = base_score;
      const int start = converse->offset[u], end = converse->offset[u + 1];

      for (int j = start; j < end; j++) {
        const int v = converse->m[j].v;
        const int w = converse->m[j].w;
        sum += w * pr_normalized[v];
      }

      pr_new[u] = sum;
      if (fabs(sum - pr[u]) > eps) {
        next_frontier[next_frontier_size++] = u;
      }
    }

    double *tmp = pr_new; pr_new = pr; pr = tmp;
    int *tmp_l = frontier; frontier = next_frontier; next_frontier = tmp_l;
    frontier_size = next_frontier_size;

    if (frontier_size == 0)
      break;
  }

  free(pr_new);
  free(pr_normalized);
  free(out_w);
  free(frontier);
  free(next_frontier);
  return pr;
}
