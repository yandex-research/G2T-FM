import random
import networkx as nx
from collections import deque, defaultdict
from math import log
from sklearn.decomposition import PCA

import dgl
import numpy as np
import torch

# >>> utils

def convert_to_adj_matrix(
        graph: dgl.DGLGraph
) -> list[list[str]]:
    adj = [[] for _ in range (graph.num_nodes())]
    src, dst = graph.edges()
    for u, v in zip(src, dst):
        if u >= v: continue
        adj[u].append(v.item())
        adj[v].append(u.item())
    return adj

class QueueWithoutDuplicates:
    def __init__(self):
        self._queue = deque()
        self._set = set()
        
    def append(self, item):
        if item not in self._set:
            self._queue.append(item)
            self._set.add(item)
            
    def popleft(self):
        item = self._queue.popleft()
        self._set.remove(item)
        return item
    
    def __len__(self):
        return len(self._queue)
    

def adjust_alpha(alpha_orig: float = 0.15) -> float:
    return alpha_orig / (2 - alpha_orig)

# <<< utils

# >>> Misc Encodings

def compute_degree_encoding(
        graph: dgl.DGLGraph,
        log: bool = True,
) -> np.ndarray:
    degrees = 0.5 * (graph.in_degrees() + graph.out_degrees())
    if log:
        degrees = torch.log(1 + degrees)
    return degrees[..., None]


@torch.no_grad()
def compute_pagerank(
        graph: dgl.DGLGraph,
        alpha=0.85,
        max_iterations=100,
        tol=1e-6,
        train_index=None,
        log: bool = True
):
    assert alpha > 0.5, "Please make sure that you provde alpha, not 1-alpha"
    g = graph
    # Initialize node features
    n_nodes = g.num_nodes()
    pv = torch.ones(n_nodes) / n_nodes
    degrees = g.out_degrees().float()
    
    # Personalization vector (uniform)
    if train_index is None:
        reset_prob = (1 - alpha) / n_nodes
    else:
        reset_prob = torch.zeros(n_nodes)
        reset_prob[train_index] = 1
        reset_prob /= reset_prob.sum()
        reset_prob *= 1 - alpha
    
    for _ in range(max_iterations):
        # Save old PageRank values
        prev_pv = pv.clone()
        
        # Message passing
        pv = dgl.ops.copy_u_sum(g, pv / degrees)
        
        # Update PageRank scores
        pv = alpha * pv + reset_prob
        
        # Check convergence
        err = torch.abs(pv - prev_pv).sum()
        if err < tol:
            break

    if log:
        pv = torch.log(tol + pv)
    return pv[..., None]

# <<< Misc Encodings
