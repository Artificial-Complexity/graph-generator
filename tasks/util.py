import networkx as nx
import matplotlib.pyplot as plt
import random
import pulp
import os
from matplotlib.animation import FuncAnimation

def generate_random_graph(num_nodes, edge_probability):
    G = nx.Graph()
    G.add_nodes_from(range(1, num_nodes + 1))
    for i in range(1, num_nodes + 1):
        for j in range(i + 1, num_nodes + 1):
            if random.random() < edge_probability:
                G.add_edge(i, j)
    return G

def visualize_graph(G, title, selected_nodes=None):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500)
    if selected_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=selected_nodes, node_color='red', node_size=500)
    plt.title(title)
    plt.show()

def visualize_graph_and_save(G, title, selected_nodes=None, folder='graph_images', step_num=0):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500)
    if selected_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=selected_nodes, node_color='red', node_size=500)
    plt.title(title)
    plt.savefig(f"{folder}/step_{step_num}.png")
    plt.close()

def mis(G, depth=0, visualize=False):
    def find_mis_degree2(G):
        mis = set()
        visited = set()
        for node in G.nodes():
            if node not in visited:
                mis.add(node)
                visited.add(node)
                visited.update(G.neighbors(node))
        return mis

    if len(G) == 0:
        return set()

    indent = "  " * depth
    if visualize:
        visualize_graph(G, f"{indent}Input Graph G (Depth: {depth})")

    if max(dict(G.degree()).values()) <= 2:
        mis_set = find_mis_degree2(G)
        if visualize:
            visualize_graph(G, f"{indent}Case 1: G has maximum degree at most 2\nMIS size: {len(mis_set)}", mis_set)
        return mis_set

    degree_one_nodes = [n for n, d in G.degree() if d == 1]
    if degree_one_nodes:
        v = degree_one_nodes[0]
        G_minus_N_v = G.copy()
        G_minus_N_v.remove_nodes_from(list(G.neighbors(v)) + [v])
        if visualize:
            visualize_graph(G, f"{indent}Case 2: Node {v} has degree 1", [v])
        return {v} | mis(G_minus_N_v, depth + 1, visualize)

    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        G1 = G.subgraph(components[0])
        G_minus_G1 = G.copy()
        G_minus_G1.remove_nodes_from(components[0])
        if visualize:
            visualize_graph(G, f"{indent}Case 3: G is not connected", components[0])
        return mis(G1, depth + 1, visualize) | mis(G_minus_G1, depth + 1, visualize)

    max_degree_node = max(G.degree(), key=lambda x: x[1])[0]
    G_minus_N_v = G.copy()
    G_minus_N_v.remove_nodes_from(list(G.neighbors(max_degree_node)) + [max_degree_node])
    G_minus_v = G.copy()
    G_minus_v.remove_node(max_degree_node)
    
    if visualize:
        visualize_graph(G, f"{indent}Case 4: Selected node {max_degree_node} with maximum degree", [max_degree_node])
    
    set1 = {max_degree_node} | mis(G_minus_N_v, depth + 1, visualize)
    set2 = mis(G_minus_v, depth + 1, visualize)
    
    final_set = set1 if len(set1) > len(set2) else set2
    
    if visualize and depth == 0:
        visualize_graph(G, "Final Graph with Maximum Independent Set", final_set)
    
    return final_set

def global_maximum_independent_set(G, visualize=False):
    prob = pulp.LpProblem("MaximumIndependentSet", pulp.LpMaximize)
    
    # Create binary variables for each node
    x = pulp.LpVariable.dicts("node", G.nodes(), cat=pulp.LpBinary)
    
    # Objective: Maximize the number of selected nodes
    prob += pulp.lpSum(x[i] for i in G.nodes())
    
    # Constraints: No two adjacent nodes can be in the set
    for (u, v) in G.edges():
        prob += x[u] + x[v] <= 1
    
    # Solve the problem
    prob.solve()
    
    # Get the nodes in the maximum independent set
    independent_set = [node for node in G.nodes() if x[node].value() > 0.5]
    
    if visualize:
        visualize_graph(G, "Final Graph with Maximum Independent Set", independent_set)
    
    return independent_set

# Additional functions to generate specific types of graphs

def generate_complete_graph(num_nodes):
    return nx.complete_graph(num_nodes)

def generate_path_graph(num_nodes):
    return nx.path_graph(num_nodes)

def generate_cycle_graph(num_nodes):
    return nx.cycle_graph(num_nodes)

def generate_star_graph(num_nodes):
    return nx.star_graph(num_nodes - 1)

def generate_tree_graph(num_nodes):
    return nx.random_tree(num_nodes)

def generate_grid_graph(dim1, dim2):
    return nx.grid_2d_graph(dim1, dim2)

import networkx as nx
import matplotlib.pyplot as plt
import os
from pathlib import Path
import datetime

def mis(G, visualize=False, step_by_step=False, run_name=None, verbose=True):
    output_dir = None
    if visualize:
        if run_name is None:
            run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("graph_images") / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if step_by_step:
            for file in output_dir.glob("step_*.png"):
                file.unlink()
    
    step_count = {}
    total_steps = 0
    step_explanations = {
        1: "Checking if the graph has maximum degree at most 2",
        2: "Graph has maximum degree at most 2, finding MIS",
        3: "Checking for nodes with degree 1",
        4: "Found a node with degree 1, including it in MIS",
        5: "Checking if the graph is connected",
        6: "Graph is not connected, processing a connected component",
        7: "Combining results from connected components",
        8: "Graph is connected with no degree 1 nodes",
        9: "Selecting a node with maximum degree",
        10: "Branching: including or excluding the max degree node"
    }
    
    # Generate a fixed layout for the entire graph only if visualizing
    pos = nx.spring_layout(G) if visualize else None
    
    def count_step(line, G=None, highlight_nodes=None, removed_nodes=None, depth=0, current_mis=None):
        nonlocal total_steps
        step_count[line] = step_count.get(line, 0) + 1
        total_steps += 1
        if verbose:
            print(f"Step {total_steps}: Executing line {line} (Depth: {depth})")
        if visualize and step_by_step and G:
            visualize_step(G, total_steps, line, highlight_nodes, removed_nodes, depth, current_mis)

    def visualize_step(G, step_number, line_number, highlight_nodes=None, removed_nodes=None, depth=0, current_mis=None):
        plt.figure(figsize=(12, 8))
        current_pos = {node: pos[node] for node in G.nodes()}
        nx.draw_networkx_edges(G, current_pos)
        nx.draw_networkx_nodes(G, current_pos, node_color='lightblue')
        if highlight_nodes:
            nx.draw_networkx_nodes(G, current_pos, nodelist=highlight_nodes, node_color='yellow')
        if removed_nodes:
            nx.draw_networkx_nodes(G, current_pos, nodelist=removed_nodes, node_color='gray', node_shape='s')
        if current_mis:
            nx.draw_networkx_nodes(G, current_pos, nodelist=list(current_mis.intersection(G.nodes())), node_color='red')
        nx.draw_networkx_labels(G, current_pos)
        
        plt.title(f"Step {step_number}: Line {line_number} (Depth: {depth})")
        plt.text(0.5, -0.1, step_explanations[line_number], ha='center', va='center', transform=plt.gca().transAxes, wrap=True)
        plt.text(0.5, -0.15, f"Current MIS size: {len(current_mis) if current_mis else 0}", ha='center', va='center', transform=plt.gca().transAxes)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_dir / f"step_{step_number:03d}.png", bbox_inches='tight')
        plt.close()

    def maximum_independent_set_degree2(G):
        independent_set = set()
        remaining_nodes = set(G.nodes())
        
        while remaining_nodes:
            v = remaining_nodes.pop()
            independent_set.add(v)
            remaining_nodes -= set(G.neighbors(v))
        
        return independent_set

    def mis_recursive(G, depth=0, current_mis=None):
        if current_mis is None:
            current_mis = set()

        # Handle empty graph
        if len(G) == 0:
            return current_mis

        count_step(1, G, depth=depth, current_mis=current_mis)
        max_degree = max(dict(G.degree()).values()) if G.degree() else 0
        if max_degree <= 2:
            count_step(2, G, depth=depth, current_mis=current_mis)
            result = maximum_independent_set_degree2(G)
            current_mis |= result
            return current_mis
        
        count_step(3, G, depth=depth, current_mis=current_mis)
        degree_one_nodes = [n for n, d in G.degree() if d == 1]
        if degree_one_nodes:
            v = degree_one_nodes[0]
            count_step(4, G, highlight_nodes=[v], removed_nodes=list(G.neighbors(v)), depth=depth, current_mis=current_mis)
            current_mis.add(v)
            return mis_recursive(G.subgraph(set(G.nodes()) - set(G.neighbors(v)) - {v}), depth+1, current_mis)
        
        count_step(5, G, depth=depth, current_mis=current_mis)
        if not nx.is_connected(G):
            count_step(6, G, depth=depth, current_mis=current_mis)
            components = list(nx.connected_components(G))
            G1 = G.subgraph(components[0])
            count_step(7, G, highlight_nodes=list(G1.nodes()), depth=depth, current_mis=current_mis)
            mis_recursive(G1, depth+1, current_mis)
            return mis_recursive(G.subgraph(set(G.nodes()) - set(G1.nodes())), depth+1, current_mis)
        
        count_step(8, G, depth=depth, current_mis=current_mis)
        count_step(9, G, depth=depth, current_mis=current_mis)
        max_degree_node = max(G.degree(), key=lambda x: x[1])[0]
        count_step(10, G, highlight_nodes=[max_degree_node], depth=depth, current_mis=current_mis)
        
        # Branch 1: Include max_degree_node
        include_mis = current_mis.copy()
        include_mis.add(max_degree_node)
        include_result = mis_recursive(G.subgraph(set(G.nodes()) - set(G.neighbors(max_degree_node)) - {max_degree_node}), depth+1, include_mis)
        
        # Branch 2: Exclude max_degree_node
        exclude_result = mis_recursive(G.subgraph(set(G.nodes()) - {max_degree_node}), depth+1, current_mis.copy())
        
        if len(include_result) > len(exclude_result):
            return include_result
        else:
            return exclude_result

    mis_result = mis_recursive(G)
    
    if verbose:
        print("\nStep counts:")
        for line, count in step_count.items():
            print(f"Line {line}: {count} times")
        print(f"Total steps: {total_steps}")
    
    if visualize:
        visualize_graph(G, mis_result, output_dir)
    
    return mis_result

def visualize_graph(G, mis_result, output_dir):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue')
    nx.draw_networkx_nodes(G, pos, nodelist=list(mis_result), node_color='red')
    nx.draw_networkx_labels(G, pos)
    plt.title("Graph with Maximum Independent Set (in red)")
    plt.text(0.5, -0.05, f"MIS size: {len(mis_result)}", ha='center', va='center', transform=plt.gca().transAxes)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / "final_graph.png", bbox_inches='tight')
    plt.close()

def generate_run_name(G):
    node_count = G.number_of_nodes()
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"nodes_{node_count}_{current_time}"

# # Example usage

# run_name = generate_run_name(G)
# result = mis(G, visualize=False, step_by_step=False, run_name=run_name, verbose=False)
# print(f"\nMaximum Independent Set: {result}")
# print(f"Size of Maximum Independent Set: {len(result)}")