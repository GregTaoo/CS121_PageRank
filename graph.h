#pragma once

typedef struct Edge {
  int u;
  int v;
  int w;
} edge;

struct Graph {
  int v;
  int e;
  int *offset;
  struct Edge *m;
};

typedef struct Graph graph;

int compare_edge(const void *x, const void *y);
void sort_graph_edges(graph* g);
graph* read_graph_file(char *filename);
void free_graph(graph* g);
int random_source_vertex(const graph* g);