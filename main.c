#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "graph.h"
#include "pagerank_omp.h"
#include "pagerank_omp_balanced.h"
#include "pagerank_omp_approx.h"
#include "pagerank_serial.h"
#include "pagerank_serial_approx.h"
#include "util.h"

#define DAMPING 0.85
#define EPS 1e-6
#define MAX_ITER 1000000

double run_serial(const graph *g, const graph *converse, char **url_map, const int mode, bool verbose) {
  double *pr = malloc(sizeof(double) * g->v);

  if (verbose) {
    printf("Serial started\n");
  }
  const double start_time = omp_get_wtime();

  if (mode < 2)
    pr = pagerank_serial(g, DAMPING, EPS, MAX_ITER, pr);
  else
    pr = pagerank_serial_approx(g, converse, DAMPING, EPS / 100, MAX_ITER, pr);

  const double total_time = omp_get_wtime() - start_time;
  if (verbose) {
    print_top_k_pr(pr, url_map, g->v, 5);
    printf("Serial time: %f\n", total_time);
  }

  free(pr);

  return total_time;
}

double run_omp(const graph *g, const graph *converse, char **url_map, const int num_threads, const int mode, bool verbose) {
  double *pr = malloc(sizeof(double) * g->v);

  if (verbose) {
    printf("Parallel started\n");
  }
  const double start_time = omp_get_wtime();

  if (mode == 0)
    pr = pagerank_omp(num_threads, g, converse, DAMPING, EPS, MAX_ITER, pr);
  else if (mode == 1)
    pr = pagerank_omp_balanced(num_threads, g, converse, DAMPING, EPS, MAX_ITER, pr);
  else
    pr = pagerank_omp_approx(num_threads, g, converse, DAMPING, EPS / 100, MAX_ITER, pr);

  const double total_time = omp_get_wtime() - start_time;
  if (verbose) {
    print_top_k_pr(pr, url_map, g->v, 5);
    printf("Parallel time: %f\n", total_time);
  }

  free(pr);

  return total_time;
}

void benchmark() {
  static const char *data_files[] = {
    "data/web-Google.mtx",
    "data/web-Stanford.mtx",
    "data/web-ShanghaiTech.mtx",
    "data/roadNet-CA.mtx",
    "data/soc-LiveJournal1.mtx",
    "data/com-orkut.ungraph.mtx"
  };
  // static const int threads[] = {0, 1, 8, 15, 22, 29, 36, 43, 50, 57, 64};
  static const int threads[] = {0, 1, 8, 15, 22, 29, 36, 43, 50, 57, 64, 65, 72, 79, 86, 93, 100, 107, 114, 121, 128};
  for (int graph_id = 0; graph_id < (int) (sizeof(data_files) / sizeof(char*)); graph_id++) {
    graph *g = read_graph_file(data_files[graph_id]);
    graph *converse = build_converse_digraph(g);
    for (int i = 0; i < (int) (sizeof(threads) / sizeof(int)); i++) {
      for (int mode = 0; mode <= 2; mode++) {
        const int num_threads = threads[i];
        double time = 0;
        for (int j = 0; j < 20; j++) {
          time += num_threads == 0 ? run_serial(g, converse, NULL, mode, false)
                                   : run_omp(g, converse, NULL, num_threads, mode, false);
        }
        time /= 20;
        printf("Dataset %s, %d Threads, Mode %d: %lfs\n", data_files[graph_id], num_threads, mode, time);
      }
    }
    free_graph(converse);
    free_graph(g);
  }
}

int main(const int argc, char **argv) {
  if (argc == 1) {
    benchmark();
    return 0;
  }

  if (!(argc == 5 || argc == 6)) {
    printf("Usage: pagerank <input_file> <num_threads> <repeat> <0: dynamic; 1: balanced; 2: approx> [url_map_file]\n");
  }
  srandom((unsigned) time(NULL));

  const char *input_file = argv[1];
  const int num_threads = atoi(argv[2]);
  const int repeat = atoi(argv[3]);
  const int mode = atoi(argv[4]);
  const char *url_map_file = argc == 6 ? argv[5] : NULL;

  graph *g = read_graph_file(input_file);
  graph *converse = build_converse_digraph(g);
  int url_map_size = 0;
  char **url_map = url_map_file != NULL ? read_url_map_file(url_map_file, &url_map_size) : NULL;

  printf("Read graph from %s: %d vertices, %d edges\n", input_file, g->v, g->e);

  double serial_time = 0, parallel_time = 0;
  for (int i = 0; i < repeat; i++) {
    serial_time += run_serial(g, converse, url_map, mode, true);
    parallel_time += run_omp(g, converse, url_map, num_threads, mode, true);
  }
  serial_time /= repeat;
  parallel_time /= repeat;
  printf("\nSerial time: %f\n", serial_time);
  printf("%s Parallel time: %f\n", mode == 0 ? "(Dynamic)" : mode == 1 ? "(Manual)" : "(Approx)", parallel_time);
  printf("Speedup: %f\n", serial_time / parallel_time);

  free_url_map(url_map, url_map_size);
  free_graph(converse);
  free_graph(g);
}
