#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "graph.h"

double *pagerank_serial(const graph *g, const double damping, const double eps, const int max_iter, double *pr) {
  const int n = g->v;

  double *pr_new = malloc(sizeof(double) * n);
  double *out_w  = malloc(sizeof(double) * n);
  memset(out_w, 0, sizeof(double) * n);

  for (int i = 0; i < n; i++)
    pr[i] = 1.0 / n;

  for (int u = 0; u < n; u++) {
    for (int i = g->offset[u]; i < g->offset[u + 1]; i++) {
      out_w[u] += g->m[i].w;
    }
  }

  for (int iter = 0; iter < max_iter; iter++) {
    for (int i = 0; i < n; i++)
      pr_new[i] = (1.0 - damping) / n;

    double dangling_sum = 0.0;

    for (int u = 0; u < n; u++) {
      if (out_w[u] == 0) {
        // dangling node
        dangling_sum += pr[u];
        continue;
      }

      for (int i = g->offset[u]; i < g->offset[u + 1]; i++) {
        const int v = g->m[i].v;
        const double w = g->m[i].w;
        pr_new[v] += damping * pr[u] * (w / out_w[u]);
      }
    }

    const double dangling_contrib = damping * dangling_sum / n;
    for (int i = 0; i < n; i++)
      pr_new[i] += dangling_contrib;

    double diff = 0.0;
    for (int i = 0; i < n; i++) {
      diff += fabs(pr_new[i] - pr[i]);
      pr[i] = pr_new[i];
    }

    if (diff < eps) {
      // printf("Converged at iter %d\n", iter);
      break;
    }
  }

  free(pr_new);
  free(out_w);
  return pr;
}
