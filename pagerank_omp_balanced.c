#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "graph.h"

int lower_bound(const int *a, int left, int right, const int value) {
  // [left, right)
  while (left < right) {
    const int mid = left + (right - left) / 2;
    if (a[mid] < value)
      left = mid + 1;
    else
      right = mid;
  }
  return left;
}

double *pagerank_omp_balanced(const int num_threads, const graph *g, const graph *converse,
                              const double damping, const double eps, const int max_iter, double *pr) {
  omp_set_num_threads(num_threads);

  const int n = g->v;
  const int e = converse->offset[n];
  const double inv_n = 1.0 / n;

  // double *pr_new        = aligned_alloc(64, sizeof(double) * n);
  // double *pr_normalized = aligned_alloc(64, sizeof(double) * n);
  double *pr_new        = malloc(sizeof(double) * n);
  double *pr_normalized = malloc(sizeof(double) * n);
  int *out_w            = malloc(sizeof(int) * n);
  memset(out_w, 0, sizeof(int) * n);

  int *start_v = malloc(sizeof(int) * (num_threads + 1));
  start_v[num_threads] = n;
#pragma omp parallel for schedule(static)
  for (int i = 0; i < num_threads; i++) {
    start_v[i] = lower_bound(converse->offset, 0, n + 1, i * e / num_threads);
    // printf("%d %d\n", i, start_v[i]);
  }

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
    double base_score;

#pragma omp parallel reduction(+: diff)
    {
      const int tid = omp_get_thread_num();

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

      double local_diff = 0.0;
      const int end_v = start_v[tid + 1];
      for (int u = start_v[tid]; u < end_v; u++) {
        double sum = base_score;
        const int start = converse->offset[u], end = converse->offset[u + 1];
        for (int i = start; i < end; ++i) {
          const int v = converse->m[i].v;
          const int w = converse->m[i].w;
          sum += w * pr_normalized[v];
        }
        pr_new[u] = sum;
        const double cur_diff = sum - pr[u];
        local_diff += fabs(cur_diff);
      }
      diff += local_diff;

// #pragma omp for schedule(dynamic, 64) reduction(+: diff)
//       for (int u = 0; u < n; ++u) {
//         double sum = base_score;
//         const int start = converse->offset[u], end = converse->offset[u + 1];
//         for (int i = start; i < end; ++i) {
//           const int v = converse->m[i].v;
//           const int w = converse->m[i].w;
//           sum += w * pr_normalized[v];
//         }
//         pr_new[u] = sum;
//         const double local_diff = sum - pr[u];
//         diff += fabs(local_diff);
//       }
    }

    double *tmp = pr_new;
    pr_new = pr;
    pr = tmp;

    if (diff < eps) {
      break;
    }
  }

  free(start_v);
  free(pr_normalized);
  free(pr_new);
  free(out_w);
  return pr;
}
