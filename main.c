#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "graph.h"
#include "pagerank_serial.h"

#define DAMPING 0.85
#define EPS 1e-6
#define MAX_ITER 10000

int main(const int argc, char **argv) {
  if (argc != 4) {
    printf("Usage: pagerank <input_file> <num_threads> <repeat>");
  }

  char *filename = argv[1];
  const int num_threads = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  srandom((unsigned) time(NULL));
  const graph *g = read_graph_file(filename);


  const double start_time = omp_get_wtime();
  double *pr = malloc(sizeof(double) * g->v);
  pagerank_serial(g, DAMPING, EPS, MAX_ITER, pr);
  const double total_time = omp_get_wtime() - start_time;
}