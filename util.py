import networkx as nx
import matplotlib.pyplot as plt
import random
import pulp

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
    return set1 if len(set1) > len(set2) else set2

def global_maximum_independent_set(G):
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
    
    # Return the nodes in the maximum independent set
    return [node for node in G.nodes() if x[node].value() > 0.5]


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