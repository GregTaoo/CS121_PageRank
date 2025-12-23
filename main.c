#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "graph.h"
#include "pagerank_omp.h"
#include "pagerank_omp_balanced.h"
#include "pagerank_serial.h"
#include "util.h"

#define DAMPING 0.85
#define EPS 1e-6
#define MAX_ITER 1000000

double run_serial(const graph *g, char **url_map) {
  double *pr = malloc(sizeof(double) * g->v);

  printf("Serial started\n");
  const double start_time = omp_get_wtime();

  pagerank_serial(g, DAMPING, EPS, MAX_ITER, pr);

  const double total_time = omp_get_wtime() - start_time;
  print_top_k_pr(pr, url_map, g->v, 5);
  printf("Serial time: %f\n", total_time);

  free(pr);

  return total_time;
}

double run_omp(const graph *g, const graph *converse, char **url_map, const int num_threads, const bool manual_load_balance) {
  double *pr = malloc(sizeof(double) * g->v);

  printf("Parallel started\n");
  const double start_time = omp_get_wtime();

  if (manual_load_balance)
    pr = pagerank_omp_balanced(num_threads, g, converse, DAMPING, EPS, MAX_ITER, pr);
  else
    pr = pagerank_omp(num_threads, g, converse, DAMPING, EPS, MAX_ITER, pr);

  const double total_time = omp_get_wtime() - start_time;
  print_top_k_pr(pr, url_map, g->v, 5);
  printf("Parallel time: %f\n", total_time);

  free(pr);

  return total_time;
}

int main(const int argc, char **argv) {
  if (!(argc == 5 || argc == 6)) {
    printf("Usage: pagerank <input_file> <num_threads> <repeat> <use_manual_load_balance> [url_map_file]");
  }
  srandom((unsigned) time(NULL));

  const char *input_file = argv[1];
  const int num_threads = atoi(argv[2]);
  const int repeat = atoi(argv[3]);
  const int use_manual_load_balance = atoi(argv[4]);
  const char *url_map_file = argc == 6 ? argv[4] : NULL;

  graph *g = read_graph_file(input_file);
  graph *converse = build_converse_digraph(g);
  int url_map_size = 0;
  char **url_map = url_map_file != NULL ? read_url_map_file(url_map_file, &url_map_size) : NULL;

  printf("Read graph from %s: %d vertices, %d edges\n", input_file, g->v, g->e);

  double serial_time = 0, parallel_time = 0;
  for (int i = 0; i < repeat; i++) {
    serial_time += run_serial(g, url_map);
    parallel_time += run_omp(g, converse, url_map, num_threads, use_manual_load_balance);
  }
  serial_time /= repeat;
  parallel_time /= repeat;
  printf("\nSerial time: %f\n", serial_time);
  printf("%s Parallel time: %f\n", use_manual_load_balance ? "(Manual)" : "(Dynamic)", parallel_time);
  printf("Speedup: %f\n", serial_time / parallel_time);

  free_url_map(url_map, url_map_size);
  free_graph(converse);
  free_graph(g);
}
