#include <stdio.h>
#include <stdlib.h>

#include "util.h"

#include <string.h>

int cmp_pr_desc(const void *a, const void *b) {
  const double x = ((NodePR*) a)->value;
  const double y = ((NodePR*) b)->value;
  if (x < y) return 1;
  if (x > y) return -1;
  return 0;
}

void print_top_k_pr(const double *pr, char **url_map, const int n, const int k) {
  NodePR *arr = malloc(sizeof(NodePR) * n);
  for (int i = 0; i < n; i++) {
    arr[i].index = i;
    arr[i].value = pr[i];
  }

  qsort(arr, n, sizeof(NodePR), cmp_pr_desc);

  for (int i = 0; i < k && i < n; i++) {
    printf("Node %d, PageRank %.6f, URL %s\n", arr[i].index, arr[i].value, url_map[arr[i].index]);
  }

  free(arr);
}

char **read_url_map_file(const char *filename, int *count) {
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Error: cannot open file %s\n", filename);
    return NULL;
  }

  char **url_map = NULL;
  *count = 0;

  char line[1024];
  while (fgets(line, sizeof(line), fp)) {
    line[strcspn(line, "\r\n")] = 0;

    if (strlen(line) == 0) continue;

    char **tmp = realloc(url_map, sizeof(char*) * (*count + 1));
    if (!tmp) {
      fprintf(stderr, "Error: realloc failed\n");
      break;
    }
    url_map = tmp;

    url_map[*count] = strdup(line);
    if (!url_map[*count]) {
      fprintf(stderr, "Error: strdup failed\n");
      break;
    }
    (*count)++;
  }

  fclose(fp);
  return url_map;
}

void free_url_map(char **url_map, const int count) {
  for (int i = 0; i < count; i++) {
    free(url_map[i]);
  }
  free(url_map);
}
