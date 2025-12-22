# An Approach to Parallelizing the PageRank Algorithm

## Abstract
The rapid expansion of the World Wide Web has necessitated efficient algorithms for information retrieval and ranking.
PageRank, the algorithm originally powering Google, measures the importance of web pages based on the graph structure of hyperlinks.
However, the sheer scale of modern web graphs—comprising billions of nodes and edges—renders serial execution computationally infeasible for real-time applications.
This project will explore the parallelization of the PageRank algorithm using OpenMP.

---

## 1. Introduction

### 1.1 Background
The World Wide Web has grown from a small network of documents into a massive, complex directed graph containing billions of nodes (pages) and edges (hyperlinks). Navigating this vast repository of information requires sophisticated search engines capable of ranking results not just by content matching, but by authority and relevance.

Before the advent of link analysis algorithms, search engines primarily relied on keyword density, which was easily manipulated. In 1998, Larry Page and Sergey Brin introduced PageRank, an algorithm that revolutionized web search by treating the web as a graph of citations. The core philosophy is recursive: a webpage is considered important if it is linked to by other important pages.

### 1.2 Motivation: Why PageRank and Parallelization?
For this programming project, I have chosen PageRank as the subject of study for two primary reasons: its industrial significance and its suitability for parallel computing.

First, **Algorithm Relevance**: PageRank is not merely a historical artifact; variants of link analysis algorithms (like HITS, TrustRank, and personalized PageRank) remain fundamental to social network analysis, recommendation systems, and bioinformatics (e.g., protein interaction networks). Understanding the mechanics of PageRank provides insight into a wide class of graph-based ranking problems.

Second, **Parallel Potential**: From a computational perspective, PageRank is an iterative application of Sparse Matrix-Vector Multiplication (SpMV).
*   **The Challenge:** A serial implementation involves iterating through millions of nodes and billions of edges sequentially. For a graph with $N$ nodes and $E$ edges, a single iteration takes $O(N+E)$ time. With real-world web graphs, where $N$ and $E$ are massive, and convergence requires many iterations, a serial program can take hours or days to complete.
*   **The Opportunity:** The algorithm exhibits high **data parallelism**. The rank calculation for a specific node (or the propagation of rank from a node) depends only on the state of the graph from the previous iteration. This independence allows us to distribute the workload across multiple processing units (cores or nodes).

Therefore, PageRank serves as an excellent benchmark for parallel programming techniques. It tests the system's ability to handle irregular memory access patterns typical of graph algorithms and allows for the demonstration of speedup and scalability using tools like OpenMP or MPI.

### 1.3 Project Goals
The objective of this project is to:
1.  Implement a baseline serial version of PageRank efficiently using Compressed Sparse Row (CSR) format.
2.  Develop a parallel version of the algorithm to utilize multi-core architecture.
3.  Analyze the performance gain (speedup) and discuss the theoretical and practical limits of the implementation.

---

## 2. Mathematical Model

### 2.1 The Random Surfer Model
PageRank models the behavior of a "random surfer" who browses the web. At any given page, the surfer has two options:
1.  Click on one of the hyperlinks on the current page with probability $d$ (the **damping factor**).
2.  Get bored and jump to a random page in the entire web with probability $1-d$ (teleportation).

The PageRank value of a page represents the long-term probability that the random surfer is currently on that page.

### 2.2 The Formula
Mathematically, the PageRank value $PR(p_i)$ for a page $p_i$ is defined as:

$$PR(p_i) = \frac{1-d}{N} + d \sum_{p_j \in M(p_i)} \frac{PR(p_j)}{L(p_j)}$$

Where:
*   $p_i$: The page for which we are calculating the rank.
*   $N$: The total number of pages in the graph.
*   $d$: The damping factor, typically set to $0.85$. This ensures the system is an irreducible Markov chain, guaranteeing convergence.
*   $M(p_i)$: The set of pages that link *to* page $p_i$ (in-neighbors).
*   $L(p_j)$: The number of outbound links (out-degree) on page $p_j$.

The term $\frac{1-d}{N}$ represents the probability of arriving at page $p_i$ via random teleportation. The summation term represents the probability of arriving at $p_i$ by following a link from a pointing page $p_j$.

### 2.3 Handling Dangling Nodes
A significant challenge in the mathematical model is the existence of "dangling nodes"—pages with no outgoing links ($L(p_j) = 0$). In a standard random walk, these nodes act as sinks, absorbing the probability mass until the total PageRank of the system diminishes to zero.

To solve this, we assume that when a random surfer hits a dangling node, they effectively restart the walk by jumping to any page in the graph with equal probability ($1/N$).

The modified equation, incorporating dangling nodes, ensures the sum of all PageRank values remains $1.0$:

$$PR(p_i) = \frac{1-d}{N} + d \left( \sum_{p_j \in M(p_i)} \frac{PR(p_j)}{L(p_j)} \right) + \frac{d}{N} \sum_{p_k \in Dangling} PR(p_k)$$

### 2.4 Convergence Criteria
The algorithm is iterative (Power Iteration method). We start with an initial distribution (usually $1/N$ for all nodes) and repeatedly update the values. The process stops when the difference between two consecutive iterations is negligible:
$$ \sum_{i=0}^{N-1} |PR_{t+1}(i) - PR_{t}(i)| < \epsilon $$

---

## 3. Serial Implementation Analysis

To establish a baseline for performance comparison, I implemented a serial version of the PageRank algorithm in C. Below is an analysis of the data structures and algorithmic logic used in the implementation.

### 3.1 Graph Representation: Compressed Sparse Row (CSR)
Web graphs are sparse matrices; a page links to only a tiny fraction of the total web. Storing this as a standard 2D array ($N \times N$) would be impossible due to memory constraints ($O(N^2)$).

My implementation uses a variant of the **Compressed Sparse Row (CSR)** format, which is standard for high-performance graph processing.
*   `offset array`: Stores the starting index of edges for each node.
*   `m array`: Stores the destination nodes and edge weights.
    This structure reduces space complexity to $O(N + E)$ and improves CPU cache locality during traversal compared to pointer-based linked lists.

### 3.2 The Algorithm Logic: Push vs. Pull
The mathematical formula described in Section 2.2 suggests a "Pull" approach: for a node $p_i$, we look at incoming neighbors to calculate its score. However, my serial implementation utilizes a **"Push" (Source-centric)** approach, which is often more intuitive when iterating over a CSR structure defined by out-edges.

**Key Steps in `pagerank_serial`:**

1.  **Preprocessing:**
    Before the main loop, the code calculates `out_w[u]`, the sum of weights of outgoing edges for every node $u$. This avoids recomputing the denominator $L(p_j)$ in every iteration.

2.  **Handling Dangling Nodes (Dynamic Accumulation):**
    Instead of a separate pass to find dangling nodes, the algorithm detects them during the main traversal.
    ```c
    if (out_w[u] == 0) {
      dangling_sum += pr[u];
      continue;
    }
    ```
    If a node has no outgoing edges, its current rank is added to a global `dangling_sum`.

3.  **The Push Update:**
    The core nested loop iterates through every source node $u$ and "pushes" its contribution to its neighbors $v$.
    ```c
    const double contribution = damping * pr[u] * (w / out_w[u]);
    pr_new[v] += contribution;
    ```
    This differs from the formula's direct translation but achieves the same mathematical result.

4.  **Global Distribution:**
    After processing all edges, the accumulated `dangling_sum` is multiplied by the damping factor and distributed evenly across all nodes (`dangling_contrib`). This, combined with the teleportation probability ($\frac{1-d}{N}$), constitutes the base value for the next iteration.

5.  **Termination:**
    The algorithm computes the L1 norm of the difference between the old and new rank vectors. If this `diff` is smaller than the threshold `eps`, the loop breaks, indicating convergence.

This serial implementation provides a correct and memory-efficient solution but suffers from performance bottlenecks on large graphs due to the single-threaded execution of the heavy loop over $N$ nodes and $E$ edges.

---

## 4. Parallel Implementation Strategy

*(**To be completed:** This section is where you will describe your solution. You need to write about 400-600 words here.)*

**Guidance for writing this section:**
*   **Approach:** State clearly that you used (e.g., OpenMP, MPI, or Pthreads).
*   **Decomposition:** How did you split the work? (e.g., Domain decomposition, splitting the `u` loop among threads).
*   **Handling Race Conditions:** This is crucial. Since your serial code uses a **Push** method (`pr_new[v] += ...`), multiple threads might try to update the same `v` at the same time. How did you solve this?
    *   *Option A:* Did you use Atomic operations (`#pragma omp atomic`)?
    *   *Option B:* Did you use arrays of locks?
    *   *Option C (Advanced):* Did you switch the algorithm to a **Pull** method to avoid write conflicts entirely?
*   **Load Balancing:** Did you use `schedule(dynamic)`? Why? (Explain that some pages have many links, some have few, so static scheduling might be unbalanced).

---

## 5. Performance Evaluation and Analysis

*(**To be completed:** This section describes your results. You need to write about 400-600 words here.)*

**Guidance for writing this section:**
*   **Environment:** List your CPU model, RAM, OS, and Compiler (gcc version).
*   **Dataset:** What graph did you use? (e.g., a random graph you generated, or a real dataset like `web-Google` from SNAP). Mention the number of nodes and edges.
*   **Metrics:**
    *   **Execution Time:** Table showing time vs. Number of Threads (1, 2, 4, 8...).
    *   **Speedup:** Calculate $S_p = T_1 / T_p$. Plot a graph.
    *   **Efficiency:** Calculate $E_p = S_p / p$.
*   **Analysis:**
    *   Does the speedup scale linearly? If not, why? (Mention memory bandwidth saturation, atomic overhead, or Amdahl's Law).
    *   Did the number of iterations change? (It shouldn't).

---

## 6. Related Works

*(**To be completed:** Brief overview of other solutions. ~200 words.)*

**Guidance for writing this section:**
*   Mention **MapReduce**: How Google originally computed PageRank using distributed MapReduce.
*   Mention **Pregel/Spark GraphX**: Modern frameworks for graph processing.
*   Mention **GPU/CUDA**: That people also run this on video cards for massive parallelism.

---

## 7. Conclusion

*(**To be completed:** Summarize your project. ~150 words.)*

**Guidance for writing this section:**
*   Restate that parallelization successfully reduced the runtime.
*   Summarize the maximum speedup achieved.
*   Mention one future improvement you could make (e.g., using SIMD instructions or distributed MPI for graphs larger than RAM).

---

## References

[1] L. Page, S. Brin, R. Motwani, and T. Winograd, "The PageRank Citation Ranking: Bringing Order to the Web," Stanford Digital Library Technologies Project, 1998.
[2] A. Langville and C. Meyer, *Google's PageRank and Beyond: The Science of Search Engine Rankings*, Princeton University Press, 2006.
[3] OpenMP Architecture Review Board, "OpenMP Application Program Interface Version 5.0," 2018.