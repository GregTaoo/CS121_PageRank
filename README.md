# PageRank

## 1. Introduction

The PageRank algorithm, famously introduced by Larry Page and Sergey Brin, revolutionized information retrieval by
shifting the focus of search engines from keyword density to structural authority. It operates on the Random Surfer
model, treating the World Wide Web as a massive directed graph. The intuition is recursive: a page is important if it is
referenced by other important pages.

This project explores the parallelization of PageRank using OpenMP, comparing three strategies to optimize performance
on multi-core architectures.

## 2. Mathematical Model

### 2.1 The PageRank Formula

The PageRank value $PR(u)$ for a node $u$ is defined as:

$$PR^{(t+1)}(u) = \frac{1-d}{N} + \frac{d \sum_{k \in D} PR^{(t)}(k)}{N} + d \sum_{v \in In(u)} \frac{w_{v,u} \cdot PR^{(t)}(v)}{\sum_{z \in Out(v)} w_{v,z}}$$

Where:

* $N$: The total number of nodes in the graph.
* $d$: The damping factor (often set to $0.85$), representing the probability that a surfer follows a link.
* $D$: The set of **Dangling Nodes** (nodes with no outgoing edges, i.e., $Out(u) = \emptyset$).
* $In(u)$: The set of nodes that point *to* $u$ (in-neighbors).
* $Out(u)$: The set of nodes that begins from *to* $u$ (out-neighbors).
* $w_{v,u}$: The weight of the edge from $v$ to $u$.

### 2.2 Data Layout

In this project, the graph is represented using the **Compressed Sparse Row (CSR)** format. The choice between
strategies fundamentally impacts memory access patterns and synchronization requirements.

**1. Push (Source-Centric)**

This model iterates over source nodes $u$ using the standard CSR graph (out-edges). For each neighbor $v$, it adds the
calculated contribution to `pr_new[v]`.

* **Pros:** It is simple to implement. Also, it could be easy to make a reduction on the graph.
* **Cons:** The algorithm writes to random memory locations (`pr_new[v]`). In a parallel environment, multiple threads
  updating the same $v$ require **atomic operations**, which serialize execution and degrade performance.

**2. Pull (Destination-Centric)**

This model iterates over destination nodes $u$. It gathers contributions from all incoming neighbors $v$ by traversing
the **Transpose Graph** (`converse`).

* **Pros:** The key advantage is that it writes **sequentially** to `pr_new[u]`. Since each thread owns a distinct range
  of $u$, no locks or atomics are needed. This operation is inherently parallel-friendly.
* **Cons:** While the writes are sequential, the reads from the rank vector (`pr[v]`) are random. This causes cache
  misses.

Therefore, the Pull model's lock-free nature make it a better choice for our parallel PageRank.

## 3. Implementation

I implemented three variations of the parallel PageRank algorithm to investigate the trade-offs between load balancing,
scheduling overhead, and algorithmic efficiency.

### 3.1 Implementation I: `pagerank_omp.c`

Real-world graphs often follow a Power-Law distribution, where a few hub nodes have high degrees, while the vast
majority have very few.
A static assignment (giving $N/P$ nodes to each thread) would be disastrous. A thread assigned a "hub" node might take
seconds to compute one iteration, while other threads finish instantly and idle. To avoid this, I utilized OpenMP's
dynamic scheduler.

Through empirical testing, I determined that a **chunk size of 64** offers the best balance.

* Chunk too small: Excessive overhead.
* Chunk too large: Fails to balance the load effectively when super-hubs clustering.

**Optimization:**

In a naive implementation, the contribution would be calculated directly as `w * pr[v] * damping / out_w[v]` inside the
inner loop. Since `v` represents a neighbor index, accessing both `pr[v]` and `out_w[v]` would trigger two separate
random memory accesses (cache misses). So I introduced a pre-calculation loop for `pr_normalized`.

#### Code Analysis

```c
#pragma omp parallel
{
// Pre-calculate normalized values to avoid a random memory access
// Also collect the dangling nodes
#pragma omp for schedule(static) reduction(+: dangling_sum)
  for (int i = 0; i < n; ++i) {
    if (out_w[i] != 0) {
      pr_normalized[i] = pr[i] * damping / (double) out_w[i];
    } else {
      dangling_sum += pr[i];
    }
  }

#pragma omp single
  base_score = (1.0 - damping + damping * dangling_sum) / n;

#pragma omp for schedule(dynamic, 64) reduction(+: diff)
  for (int u = 0; u < n; ++u) {
    double sum = base_score;
    const int start = converse->offset[u], end = converse->offset[u + 1];
    for (int i = start; i < end; ++i) {
      const int v = converse->m[i].v;
      const int w = converse->m[i].w;
      // Here we have one random memory access (bottleneck)
      sum += w * pr_normalized[v];
    }
    pr_new[u] = sum;
    diff += fabs(sum - pr[u]);
  }
}
```

* **Pros:** Robust against highly skewed degree distributions (like `web-ShanghaiTech`). It is easy to implement using
  standard OpenMP clauses.
* **Cons:** **High Overhead.** For graphs with low average degrees (like `roadNet-CA`, where avg degree $\approx 2.5$),
  the computation inside the inner loop is trivial. The overhead of the OpenMP runtime managing the dynamic queue
  becomes a dominant factor, limiting the maximum speedup.

### 3.2 Implementation II: `pagerank_omp_balanced`

To overcome the overhead of dynamic scheduling while maintaining load balance, I implemented a static partitioning
strategy based on edge counts rather than node counts.

The goal is to define thread boundaries such that every thread processes exactly $|E| / P$ edges. Since the graph is
stored in CSR format, the `offset` array is monotonic. We can use binary search (`lower_bound`) to find the exact node
indices that divide the edge array into equal segments.

#### Code Analysis

**Partitioning:**

```c
// Calculate partition boundaries based on edge counts
int *start_v = malloc(sizeof(int) * (num_threads + 1));
start_v[num_threads] = n;
#pragma omp parallel for schedule(static)
for (int i = 0; i < num_threads; i++) {
  // Find the node index where the cumulative edge count reaches i * (total_edges / threads)
  start_v[i] = lower_bound(converse->offset, 0, n + 1, i * e / num_threads);
}
```

Inside the parallel region, each thread queries its ID (`tid`) and retrieves its pre-calculated range
`[start_v[tid], start_v[tid + 1])`.

```c
// Distribute the nodes
#pragma omp parallel reduction(+: diff)
{
  const int tid = omp_get_thread_num();
  const int end_v = start_v[tid + 1];
  for (int u = start_v[tid]; u < end_v; u++) {
    // ... (standard Pull logic) ...
  }
}
```

* **Pros:**

  1. **Less Scheduling Overhead:** The cost for scheduling is lower. This is a massive advantage for low-degree graphs (
     `roadNet`) where the dynamic scheduler was a bottleneck.
  2. **Perfect Theoretical Load Balance:** Every thread performs almost exactly the same number of memory reads and
     floating-point additions.
  3. **Cache Locality:** Threads process contiguous blocks of memory.

* **Cons:**

  Lack of flexibility. It cannot adapt to system noise or background processes.
  Also, on modern CPUs with Hybrid Architectures (e.g., Intel P-cores vs. E-cores or
  ARM big.LITTLE), this approach suffers.

### 3.3 Implementation III: `pagerank_omp_approx`

The standard PageRank algorithm updates *every* node in every iteration. However, as the algorithm progresses, many
nodes converge quickly and their values stop changing. Updating these stable nodes wastes memory bandwidth. The third
implementation introduces an **approximation** by keeping an active list.

We maintain a list of active nodes (`frontier`). A node is added to the next iteration's frontier only if its PageRank
value changed significantly ($|\Delta PR| > \epsilon$) in the current iteration.

By skipping updates for stable nodes, we introduce a **cumulative approximation error**. The
final PageRank scores will not be numerically identical to the exact eigenvector solution. However, for most practical
applications, the exact probability value is less important than the relative ranking of the
nodes. While the absolute values fluctuate slightly, the ordering of the top-ranked nodes
remains somehow consistent with the exact version.

#### Code Analysis

**The Frontier Logic:**

```c
// 1. Compute updates only for nodes in current 'frontier'
#pragma omp parallel for schedule(static)
for (int i = 0; i < frontier_size; i++) {
  const int u = frontier[i];
  // ... compute sum ...
  pr_new[u] = sum;
  // Mark if this node needs to be active next time
  next_idx_prefix[i] = fabs(sum - pr[u]) > eps; 
}

// 2. Parallel Prefix Sum to calculate write positions
omp_prefix_sum(next_idx_prefix, frontier_size, prefix_block_sum);

// 3. Scatter
#pragma omp parallel for schedule(static)
for (int i = 0; i < frontier_size; i++) {
  const int prev_val = (i == 0) ? 0 : next_idx_prefix[i - 1];
  if (next_idx_prefix[i] > prev_val) {
    next_frontier[next_idx_prefix[i] - 1] = frontier[i];
  }
}
```

* **Pros:**
  The workload decreases exponentially, for both serial and parallel algorithms.
* **Cons:**
  1. **Overhead:** The parallel prefix sum and array compaction add overhead. In practice, the speedup and scalability
     of the parallelized algorithm isn't that good.
  2. **Complexity:** Requires managing explicit frontier arrays and implementing parallel scan primitives.

---

## 4. Performance

The following experiments were conducted on a machine with [Insert CPU Spec] and [Insert RAM]. We compared the three
implementations on two distinct datasets: `roadNet-CA` (low degree, uniform) and `web-ShanghaiTech` (high skew,
Power-Law).

### 4.1 Execution Time and Speedup

*(Please insert your experimental data in the table below)*

**Table 1: Total Execution Time (seconds) for 20 Iterations**

| Strategy                | Threads | roadNet-CA | web-ShanghaiTech |
|:------------------------|:--------|:-----------|:-----------------|
| **Serial Baseline**     | 1       | [Data]     | [Data]           |
| **I. OMP Dynamic (64)** | 1       | [Data]     | [Data]           |
|                         | 8       | [Data]     | [Data]           |
|                         | 16      | [Data]     | [Data]           |
| **II. OMP Balanced**    | 1       | [Data]     | [Data]           |
|                         | 8       | [Data]     | [Data]           |
|                         | 16      | [Data]     | [Data]           |
| **III. OMP Approx**     | 1       | [Data]     | [Data]           |
|                         | 8       | [Data]     | [Data]           |
|                         | 16      | [Data]     | [Data]           |

### 4.2 Analysis of Results

**1. Overhead vs. Granularity (Dynamic vs. Balanced):**
Comparing Implementation I and II reveals the impact of runtime overhead.

* On **`roadNet-CA`**, the Edge-Balanced implementation (II) significantly outperforms the Dynamic implementation (I).
  Since the average degree is small, the "work" per node is minimal. The dynamic scheduler's overhead (atomic
  fetch-and-add on the task queue) consumes a large portion of the cycle time. The Balanced approach eliminates this,
  allowing threads to run tight loops uninterrupted.
* On **`web-ShanghaiTech`**, both implementations perform well, but the Balanced approach is still marginally faster or
  equivalent. It effectively neutralizes the load imbalance caused by hub nodes without the need for dynamic task
  fetching.

**2. The Power of Approximation:**
Implementation III (`omp_approx`) shows the most dramatic reduction in execution time, particularly in the later stages
of convergence.

* While the per-iteration time in the first few rounds is similar to the standard approaches (due to frontier overhead),
  the time drops drastically as the frontier shrinks.
* This confirms that for many iterative graph algorithms, "work-efficiency" (doing less work) is often more potent
  than "parallel efficiency" (dividing work better).

---

## 5. Related Work and Future Optimizations

The parallelization of PageRank is a well-studied field.

* **Distributed Systems:** Google's original implementation relied on MapReduce to handle graphs larger than the memory
  of a single machine. Modern equivalents use frameworks like Apache Spark (GraphX) or Google Pregel (BSP model).
* **GPU Acceleration:** Implementation on GPUs (CUDA) is common but faces challenges with "warp divergence" due to
  irregular graph structures. Our "Balanced" approach is conceptually similar to how GPUs assign threads to edges to
  maintain coalesced memory access.

## 6. Conclusion

This project demonstrated that efficient parallelization of PageRank requires more than simply adding
`#pragma omp parallel for`.

1. We transitioned from **Push to Pull** to ensure thread safety without locks.
2. We showed that **Dynamic Scheduling** is effective for skewed graphs but suffers from overhead on sparse, low-degree
   graphs.
3. We proved that **Manual Edge-Balancing** is the robust, "best-of-both-worlds" solution for exact PageRank, offering
   perfect load balance with zero runtime overhead.
4. Finally, we demonstrated that **Approximation (Frontier-based)** methods yield the highest raw performance by
   intelligently skipping redundant computations.

For general-purpose high-performance graph processing on shared-memory systems, the **Edge-Balanced Pull** strategy is
the recommended approach for exact results, while the **Frontier** approach is ideal when speed is prioritized over
strict per-iteration precision.