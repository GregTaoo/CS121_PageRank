#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "graph.h"

#include <stdbool.h>
#include <string.h>

int compare_edge(const void *x, const void *y) {
  const struct Edge *a = (struct Edge*) x, *b = (struct Edge*) y;
  return a->u != b->u ? a->u - b->u : a->v - b->v;
}

void sort_graph_edges(graph* g) {
  assert(g != NULL);
  assert(g->m != NULL);

  qsort(g->m, g->e, sizeof(struct Edge), compare_edge);
  g->offset = malloc(sizeof(int) * (g->v + 1));
  memset(g->offset, 0, sizeof(int) * (g->v + 1));

  int i = 0, j = 0;
  while (i < g->v) {
    g->offset[i] = j;
    while (g->m[j].u == i) {
      ++j;
    }
    ++i;
  }
  g->offset[g->v] = g->e;
}

graph* read_graph_file(const char *filename) {
  FILE *f = fopen(filename, "r");
  if (!f) {
    fprintf(stderr, "Cannot open file: %s\n", filename);
    return NULL;
  }

  char line[1024];

  if (!fgets(line, sizeof(line), f)) {
    fclose(f);
    return NULL;
  }

  int is_symmetric = 0;
  if (strstr(line, "%%MatrixMarket") == line) {
    const char *last_word = strrchr(line, ' ');
    if (last_word) {
      if (strncmp(last_word + 1, "symmetric", 9) == 0) {
        is_symmetric = 1;
      }
    }
  } else {
    fprintf(stderr, "File does not start with %%MatrixMarket header.\n");
    fclose(f);
    return NULL;
  }

  do {
    if (!fgets(line, sizeof(line), f)) {
      fclose(f);
      return NULL;
    }
  } while (line[0] == '%');

  int M, N, L;
  if (sscanf(line, "%d %d %d", &M, &N, &L) != 3) {
    fprintf(stderr, "Invalid size line.\n");
    fclose(f);
    return NULL;
  }
  assert(M == N);

  graph *g    = malloc(sizeof(graph));
  g->e        = is_symmetric ? 2 * L : L;
  g->m        = (struct Edge*) malloc(sizeof(struct Edge) * g->e);

  int u, v, w;
  for (int i = 0; i < L; i++) {
    if (fscanf(f, "%d %d %d", &u, &v, &w) != 3) {
      fprintf(stderr, "Invalid entry at line %d\n", i + 1);
      free(g->m);
      free(g);
      fclose(f);
      return NULL;
    }

    g->m[i].u = u;
    g->m[i].v = v;
    g->m[i].w = w;

    if (u > M)
      M = N = u;
    if (v > M)
      M = N = v;
    if (is_symmetric) {
      g->m[i + L].u = v;
      g->m[i + L].v = u;
      g->m[i + L].w = w;
    }
  }

  g->v = M + 1;

  fclose(f);

  // preprocess
  sort_graph_edges(g);

  assert(g != NULL);
  assert(g->m != NULL);
  assert(g->offset != NULL);
  return g;
}

graph* build_converse_digraph(const graph *g) {
  graph *converse = malloc(sizeof(graph));
  converse->v     = g->v;
  converse->e     = g->e;
  converse->m     = (struct Edge*) malloc(sizeof(struct Edge) * g->e);

  for (int i = 0; i < g->e; i++) {
    converse->m[i].u = g->m[i].v;
    converse->m[i].v = g->m[i].u;
    converse->m[i].w = g->m[i].w;
  }

  sort_graph_edges(converse);

  return converse;
}

void free_graph(graph *g) {
  free(g->offset);
  free(g->m);
  free(g);
}

int random_source_vertex(const graph *g) {
  return random() % g->v;
}
