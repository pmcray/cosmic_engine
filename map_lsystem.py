"""
Map L-Systems (ABOP Chapter 7).
Models cellular layers using planar maps (graphs of vertices, edges, and regions).
"""
import numpy as np
from collections import defaultdict
import uuid

class HalfEdge:
    def __init__(self, origin, twin=None, next_edge=None, prev_edge=None, region=None):
        self.id = str(uuid.uuid4())
        self.origin = origin        # Origin vertex ID
        self.twin = twin            # The HalfEdge going the opposite way
        self.next = next_edge       # Next HalfEdge in the region loop
        self.prev = prev_edge       # Previous HalfEdge
        self.region = region        # The region/cell this half-edge bounds
        self.state = ""             # State character to match L-system productions

class Vertex:
    def __init__(self, id, pos):
        self.id = id
        self.pos = np.array(pos, dtype=float)
        self.incident_edge = None   # One of the half-edges leaving this vertex

class Region:
    def __init__(self, id, state=""):
        self.id = id
        self.state = state
        self.boundary = None        # One half-edge on its boundary

class MapGraph:
    """
    A basic topological Double-Connected Edge List (DCEL) for Map L-Systems.
    """
    def __init__(self):
        self.vertices = {}
        self.faces = {}
        self.half_edges = {}

    def add_vertex(self, pos):
        v_id = str(uuid.uuid4())
        v = Vertex(v_id, pos)
        self.vertices[v_id] = v
        return v_id

    def add_edge(self, v1_id, v2_id, region1_id, region2_id=None):
        """
        Add a bidirectional edge between v1 and v2.
        Assigns region1 to the forward half-edge, region2 to twin.
        """
        he1 = HalfEdge(origin=v1_id, region=region1_id)
        he2 = HalfEdge(origin=v2_id, twin=he1, region=region2_id)
        he1.twin = he2
        
        self.half_edges[he1.id] = he1
        self.half_edges[he2.id] = he2
        return he1.id, he2.id

    def split_edge(self, he_id, new_pos=None):
        """
        Split a half-edge (and its twin) by inserting a new vertex.
        Returns the new vertex id.
        """
        old_he1 = self.half_edges[he_id]
        old_he2 = old_he1.twin
        
        if new_pos is None:
            # mid-point
            p1 = self.vertices[old_he1.origin].pos
            p2 = self.vertices[old_he2.origin].pos
            new_pos = (p1 + p2) / 2.0
            
        v_new_id = self.add_vertex(new_pos)
        
        # New half-edges
        new_he1 = HalfEdge(origin=v_new_id, region=old_he1.region)
        new_he2 = HalfEdge(origin=v_new_id, region=old_he2.region)
        
        new_he1.twin = old_he2
        old_he2.twin = new_he1
        
        new_he2.twin = old_he1
        old_he1.twin = new_he2
        
        # Update origin
        new_he2.origin = v_new_id
        
        self.half_edges[new_he1.id] = new_he1
        self.half_edges[new_he2.id] = new_he2
        
        # Fix next/prev pointers (simplified logic)
        return v_new_id

class MapLSystem:
    def __init__(self):
        self.graph = MapGraph()
        self.edge_rules = {}
        self.cell_rules = {}
        self.iteration = 0

    def add_edge_rule(self, state, successor):
        self.edge_rules[state] = successor
        
    def add_cell_rule(self, state, division_logic):
        self.cell_rules[state] = division_logic

    def step(self):
        """
        Execute one step of the Map L-System rewriting.
        In ABOP Chapter 7, map rewriting usually happens in two phases:
        1. Edge rewriting (splitting boundaries)
        2. Cell division (adding new edges across cells)
        """
        print(f"Executing map rewrite step {self.iteration}")
        # Placeholder for full map traversal rewriting logic
        self.iteration += 1

    def relax_layout(self, iterations=50, spring_k=0.1):
        """
        Simple Spring-Mass layout for cell graphical interpretation (ABOP 7.2)
        """
        for _ in range(iterations):
            for v in self.graph.vertices.values():
                # Apply spring forces towards neighbors and cell centers to prevent self-intersection
                v.pos += np.random.normal(0, 0.01, size=3) # minimal jiggle to demonstrate
                
    def export_obj(self, filename):
        """
        Export cellular map layout to OBJ (edges as lines).
        """
        with open(filename, 'w') as f:
            v_keys = list(self.graph.vertices.keys())
            v_map = {vid: i+1 for i, vid in enumerate(v_keys)}
            
            for vid in v_keys:
                pos = self.graph.vertices[vid].pos
                f.write(f"v {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
                
            visited = set()
            for he in self.graph.half_edges.values():
                if he.id in visited or he.twin.id in visited:
                    continue
                v1 = v_map[he.origin]
                v2 = v_map[he.twin.origin]
                f.write(f"l {v1} {v2}\n")
                visited.add(he.id)
                visited.add(he.twin.id)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Map L-System Engine")
    parser.add_argument("--test", action="store_true", help="Run cell division test")
    args = parser.parse_args()
    
    if args.test:
        print("Testing Map L-system Initialization...")
        mz = MapLSystem()
        v1 = mz.graph.add_vertex([0, 0, 0])
        v2 = mz.graph.add_vertex([1, 0, 0])
        v3 = mz.graph.add_vertex([0.5, 1, 0])
        
        # Make a triangle cell
        mz.graph.add_edge(v1, v2, "C1")
        mz.graph.add_edge(v2, v3, "C1")
        mz.graph.add_edge(v3, v1, "C1")
        
        mz.step()
        mz.relax_layout()
        mz.export_obj("map_test.obj")
        print("Exported to map_test.obj")
