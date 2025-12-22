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
  return left & ~15;
}

void pagerank_omp(const int num_threads, const graph *g, const graph *converse,
                  const double damping, const double eps, const int max_iter, double *pr) {
  omp_set_num_threads(num_threads);

  const int n = g->v;
  const int e = converse->offset[n];

  double *pr_new = aligned_alloc(64, sizeof(double) * n);
  // double *pr_new = malloc(sizeof(double) * n);
  double *out_w  = malloc(sizeof(double) * n);
  memset(out_w, 0, sizeof(double) * n);

  int *start_v = malloc(sizeof(int) * (num_threads + 1));
  start_v[num_threads] = n;
#pragma omp parallel for schedule(static)
  for (int i = 0; i < num_threads; i++) {
    start_v[i] = lower_bound(converse->offset, 0, n + 1, i * e / num_threads);
  }

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    for (int u = start_v[tid]; u < start_v[tid + 1]; u++) {
      pr[u] = 1.0 / n;
      int weight = 0;
      for (int i = g->offset[u]; i < g->offset[u + 1]; i++) {
        weight += g->m[i].w;
      }
      out_w[u] = weight;
    }
  }

// #pragma omp parallel for schedule(static)
//   for (int u = 0; u < n; u++) {
//     pr[u] = 1.0 / n;
//     for (int i = g->offset[u]; i < g->offset[u + 1]; i++) {
//       out_w[u] += g->m[i].w;
//     }
//   }
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

#pragma omp parallel reduction(+: diff, new_dangling_sum)
    {
      const int tid = omp_get_thread_num();
      double local_diff = 0.0, local_dangling_sum = 0.0;
      // printf("%d %d %d\n", tid, start_v[tid], start_v[tid + 1]);
      for (int u = start_v[tid]; u < start_v[tid + 1]; u++) {
        const int start = converse->offset[u];
        const int end = converse->offset[u + 1];
        double sum = (1.0 - damping) / n + dangling_contrib;
        for (int i = start; i < end; i++) {
          const int v = converse->m[i].v;
          const double w = converse->m[i].w;
          sum += damping * pr[v] * (w / out_w[v]);
        }
        pr_new[u] = sum;
        local_diff += fabs(sum - pr[u]);
        // dangling node
        if (out_w[u] == 0) {
          local_dangling_sum += pr_new[u];
        }
      }
      diff += local_diff;
      new_dangling_sum += local_dangling_sum;
    }

// #pragma omp parallel for schedule(dynamic, 16)
//     for (int u = 0; u < n; u++) {
//       double sum = (1.0 - damping) / n + dangling_contrib;
//       for (int i = converse->offset[u]; i < converse->offset[u + 1]; i++) {
//         const int v = converse->m[i].v;
//         const double w = converse->m[i].w;
//         sum += damping * pr[v] * (w / out_w[v]);
//       }
//       pr_new[u] = sum;
//     }
// #pragma omp parallel for schedule(static) reduction(+: diff)
//     for (int i = 0; i < n; i++) {
//       diff += fabs(sum - pr[u]);
//     }

    double *tmp = pr_new;
    pr_new = pr;
    pr = tmp;
    dangling_sum = new_dangling_sum;

    if (diff < eps) {
      // printf("Converged at iter %d\n", iter);
      break;
    }
  }

  free(start_v);
  free(pr_new);
  free(out_w);
}
