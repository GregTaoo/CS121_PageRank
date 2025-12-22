#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#include "graph.h"
#include "util.h"
#include "pagerank_serial.h"

#define DAMPING 0.85
#define EPS 1e-6
#define MAX_ITER 1000000

double run_serial(const graph *graph, char **url_map) {
  double *pr = malloc(sizeof(double) * graph->v);

  const double start_time = omp_get_wtime();

  pagerank_serial(graph, DAMPING, EPS, MAX_ITER, pr);

  const double total_time = omp_get_wtime() - start_time;
  print_top_k_pr(pr, url_map, graph->v, 15);
  printf("Total time: %f\n", total_time);

  free(pr);

  return total_time;
}

int main(const int argc, char **argv) {
  if (argc != 4) {
    printf("Usage: pagerank <input_file> <url_map_file> <num_threads> <repeat>");
  }
  srandom((unsigned) time(NULL));

  const char *input_file = argv[1];
  const char *url_map_file = argv[2];
  const int num_threads = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  graph *graph = read_graph_file(input_file);
  int url_map_size = 0;
  char **url_map = read_url_map_file(url_map_file, &url_map_size);

  printf("Read graph from %s: %d vertices, %d edges\n", input_file, graph->v, graph->e);

  run_serial(graph, url_map);

  free_url_map(url_map, url_map_size);
  free_graph(graph);
}
