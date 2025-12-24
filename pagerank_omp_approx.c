#include <math.h>
#include <omp.h>
#include <stdlib.h>

#include "graph.h"

void omp_prefix_sum(int *sum, const int n, int *block_sum) {
  int tid, i, prefix;

#pragma omp parallel default(none) shared(sum, n, block_sum) private(tid, i, prefix)
  {
    tid = omp_get_thread_num();
    prefix = 0;

#pragma omp for schedule(static)
    for (i = 0; i < n; i++) {
      prefix += sum[i];
      sum[i] = prefix;
    }
    block_sum[tid] = prefix;
    prefix = 0;

#pragma omp barrier
    for (i = 0; i < tid; i++) {
      prefix += block_sum[i];
    }

#pragma omp for schedule(static)
    for (i = 0; i < n; i++) {
      sum[i] += prefix;
    }
  }
}

double *pagerank_omp_approx(const int num_threads, const graph *g, const graph *converse,
                            const double damping, const double eps, const int max_iter, double *pr) {
  omp_set_num_threads(num_threads);

  const int n = g->v;
  const double inv_n = 1.0 / n;

  double *pr_new        = malloc(sizeof(double) * n);
  double *pr_normalized = malloc(sizeof(double) * n);
  int *out_w            = malloc(sizeof(int) * n);

  int *prefix_block_sum = malloc(sizeof(int) * (num_threads + 1));
  int *frontier         = malloc(sizeof(int) * n);
  int *next_frontier    = malloc(sizeof(int) * n);
  int *next_idx_prefix  = malloc(sizeof(int) * n);

  int frontier_size = n;

#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; i++) {
    pr[i] = inv_n;
    frontier[i] = i;
    const int start = g->offset[i], end = g->offset[i + 1];
    int w = 0;
    for (int j = start; j < end; ++j) {
      w += g->m[j].w;
    }
    out_w[i] = w;
  }

  for (int iter = 0; iter < max_iter; ++iter) {
    double dangling_sum = 0.0;

#pragma omp parallel for schedule(static) reduction(+: dangling_sum)
    for (int i = 0; i < frontier_size; i++) {
      const int u = frontier[i];
      if (out_w[u] != 0) {
        pr_normalized[u] = pr[u] * damping / (double) out_w[u];
      } else {
        dangling_sum += pr[u];
      }
    }

    const double base_score = (1.0 - damping + damping * dangling_sum) / n;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < frontier_size; i++) {
      const int u = frontier[i];
      double sum = base_score;
      const int start = converse->offset[u], end = converse->offset[u + 1];

      for (int j = start; j < end; ++j) {
        const int v = converse->m[j].v;
        const int w = converse->m[j].w;
        sum += w * pr_normalized[v];
      }

      pr_new[u] = sum;
      next_idx_prefix[i] = fabs(sum - pr[u]) > eps;
    }

    omp_prefix_sum(next_idx_prefix, frontier_size, prefix_block_sum);

    const int next_frontier_size = frontier_size > 0 ? next_idx_prefix[frontier_size - 1] : 0;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < frontier_size; i++) {
      const int prev_val = i == 0 ? 0 : next_idx_prefix[i - 1];
      if (next_idx_prefix[i] > prev_val) {
        next_frontier[next_idx_prefix[i] - 1] = frontier[i];
      }
    }

    double *tmp = pr_new; pr_new = pr; pr = tmp;
    int *tmp_l = frontier; frontier = next_frontier; next_frontier = tmp_l;
    frontier_size = next_frontier_size;

    if (frontier_size == 0)
      break;
  }

  free(prefix_block_sum);
  free(frontier);
  free(next_frontier);
  free(next_idx_prefix);
  free(pr_normalized);
  free(pr_new);
  free(out_w);

  return pr;
}