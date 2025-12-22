#pragma once

typedef struct {
  int index;
  double value;
} NodePR;

int cmp_pr_desc(const void *a, const void *b);
void print_top_k_pr(const double *pr, char **url_map, int n, int k);
char **read_url_map_file(const char *filename, int *count);
void free_url_map(char **url_map, int count);
