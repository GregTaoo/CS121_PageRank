#include <math.h>
#include <omp.h>
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

void pagerank_omp(const int num_threads, const graph *g, const graph *converse,
                  const double damping, const double eps, const int max_iter, double *pr) {
  omp_set_num_threads(num_threads);

  const int n = g->v;
  const int e = converse->offset[n];

  double *pr_new = malloc(sizeof(double) * n);
  double *out_w  = malloc(sizeof(double) * n);
  memset(out_w, 0, sizeof(double) * n);

  int *start_v = malloc(sizeof(int) * (num_threads + 1));
  start_v[num_threads] = n;
#pragma omp parallel for schedule(static)
  for (int i = 0; i < num_threads; i++) {
    start_v[i] = lower_bound(converse->offset, 0, n + 1, i * e / num_threads);
  }

#pragma omp parallel for schedule(static)
  for (int u = 0; u < n; u++) {
    pr[u] = 1.0 / n;
    for (int i = g->offset[u]; i < g->offset[u + 1]; i++) {
      out_w[u] += g->m[i].w;
    }
  }

  for (int iter = 0; iter < max_iter; iter++) {
    double dangling_sum = 0.0;

#pragma omp parallel for schedule(static) reduction(+: dangling_sum)
    for (int i = 0; i < n; i++) {
      if (out_w[i] == 0) {
        // dangling node
        dangling_sum += pr[i];
      }
    }

    const double dangling_contrib = damping * dangling_sum / n;

#pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      // printf("%d %d %d\n", tid, start_v[tid], start_v[tid + 1]);
      for (int u = start_v[tid]; u < start_v[tid + 1]; u++) {
        pr_new[u] = (1.0 - damping) / n + dangling_contrib;
        for (int i = converse->offset[u]; i < converse->offset[u + 1]; i++) {
          const int v = converse->m[i].v;
          const double w = converse->m[i].w;
          pr_new[u] += damping * pr[v] * (w / out_w[v]);
        }
      }
    }

// #pragma omp parallel for schedule(dynamic, 16)
//     for (int u = 0; u < n; u++) {
//       pr_new[u] = (1.0 - damping) / n + dangling_contrib;
//       for (int i = converse->offset[u]; i < converse->offset[u + 1]; i++) {
//         const int v = converse->m[i].v;
//         const double w = converse->m[i].w;
//         pr_new[u] += damping * pr[v] * (w / out_w[v]);
//       }
//     }

    double diff = 0.0;
#pragma omp parallel for schedule(static) reduction(+: diff)
    for (int i = 0; i < n; i++) {
      diff += fabs(pr_new[i] - pr[i]);
    }

    double *tmp = pr_new;
    pr_new = pr;
    pr = tmp;

    if (diff < eps) {
      // printf("Converged at iter %d\n", iter);
      break;
    }
  }

  free(start_v);
  free(pr_new);
  free(out_w);
}
