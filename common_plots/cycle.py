from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Set
from constant import ASSERT, CONCLUDE

BARRIERS = {ASSERT, CONCLUDE}

def get_segment_trace(states: List[str], barriers: Set[str] = BARRIERS) -> List[List[str]]:
    # Split the walk at barriers; barriers themselves are excluded from segments
    segments, current = [], []
    for s in states:
        if s in barriers:
            if current: 
                segments.append(current)
                current = []
        else:
            current.append(s)
    if current: 
        segments.append(current)
    return segments

# ---- Tarjan SCC ----
def tarjan_scc(graph_adj: Dict[str, Set[str]]) -> List[Set[str]]:
    index = 0; stack = []; on_stack = set(); indices = {}; lowlink = {}; out = []
    def strongconnect(v):
        nonlocal index
        indices[v] = lowlink[v] = index; index += 1
        stack.append(v); on_stack.add(v)
        for w in graph_adj.get(v, set()):
            if w not in indices:
                strongconnect(w); lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])
        if lowlink[v] == indices[v]:
            s = set()
            while True:
                w = stack.pop(); on_stack.remove(w); s.add(w)
                if w == v: break
            out.append(s)
    for v in graph_adj.keys():
        if v not in indices: strongconnect(v)
    return out

def build_multigraph_edges(seq: List[str]) -> List[Tuple[str, str]]:
    return list(zip(seq[:-1], seq[1:]))

def adjacency_from_edges(edges: List[Tuple[str, str]]) -> Dict[str, Set[str]]:
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
    return adj

def cyclical_scc_nodes(adj: Dict[str, Set[str]]) -> Set[str]:
    cyc = set()
    for scc in tarjan_scc(adj) if adj else []:
        if len(scc) > 1: 
            cyc |= scc   # union
        else:
            n = next(iter(scc))
            if n in adj and n in adj[n]:
                cyc.add(n)  # self-loop
    return cyc

def cycle_edge_mask(seq: List[str]) -> List[bool]:
    edges = build_multigraph_edges(seq)
    adj = adjacency_from_edges(edges)
    cyc_nodes = cyclical_scc_nodes(adj)
    return [(u in cyc_nodes and v in cyc_nodes) for (u, v) in edges]

def analyze_trace(states: List[str]):
    segments = get_segment_trace(states)
    total_edges = cyc_edges = 0
    for seg in segments:
        marks = cycle_edge_mask(seg)
        total_edges += len(marks)
        cyc_edges += sum(marks)
    frac = (cyc_edges / total_edges) if total_edges else 0.0
    # (#segments, density, #cycle_edges, #edges)
    return len(segments), frac, cyc_edges, total_edges  
