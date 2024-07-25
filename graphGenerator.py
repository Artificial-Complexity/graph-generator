import networkx as nx
import os
from pyvis.network import Network
from IPython.display import display, HTML
import matplotlib.pyplot as plt

class GraphGenerator:
    def __init__(self):
        """Initialize the ExtendedGraphGenerator with default values."""
        self.G = None
        self.graph_type = ""
        self.output_folder = "graphs"
        self.mis_folder = os.path.join(self.output_folder, "mis_solutions")
        if not os.path.exists(self.mis_folder):
            os.makedirs(self.mis_folder)

    def generate_erdos_renyi(self, n, p):
        """Generate an Erdős-Rényi graph."""
        self.G = nx.erdos_renyi_graph(n, p)
        self.graph_type = f"Erdos-Renyi_n{n}_p{p}"

    def generate_barabasi_albert(self, n, m):
        """Generate a Barabási-Albert graph."""
        self.G = nx.barabasi_albert_graph(n, m)
        self.graph_type = f"Barabasi-Albert_n{n}_m{m}"

    def generate_watts_strogatz(self, n, k, p):
        """Generate a Watts-Strogatz graph."""
        self.G = nx.watts_strogatz_graph(n, k, p)
        self.graph_type = f"Watts-Strogatz_n{n}_k{k}_p{p}"

    def generate_random_regular(self, n, d):
        """Generate a Random Regular graph."""
        self.G = nx.random_regular_graph(d, n)
        self.graph_type = f"Random-Regular_n{n}_d{d}"

    def visualize_graph_interactive(self, save=True, cdn_resources='in_line'):
        """
        Visualize the generated graph using Pyvis for an interactive HTML file.
        
        Parameters:
        save (bool): Whether to save the graph as an HTML file. Default is True.
        cdn_resources (str): The source of CDN resources. Options are 'in_line', 'remote', or 'local'. Default is 'in_line'.
        """
        if self.G is None:
            print("No graph generated yet. Please generate a graph first.")
            return

        net = Network(notebook=True, cdn_resources=cdn_resources)
        net.from_nx(self.G)
        net.show_buttons(filter_=['physics'])

        html_filename = f"{self.graph_type}_plot.html"
        html_path = os.path.join(self.output_folder, html_filename)

        if save:
            net.save_graph(html_path)
            print(f"Graph plot saved as: {html_path}")

        # Display the saved HTML file content in the Jupyter notebook
        display(HTML(html_path))

    def get_graph_info(self):
        """
        Get information about the generated graph.
        
        Returns:
        str: A string containing information about the graph.
        """
        if self.G is None:
            return "No graph generated yet."

        info = f"Graph Type: {self.graph_type}\n"
        info += f"Number of nodes: {self.G.number_of_nodes()}\n"
        info += f"Number of edges: {self.G.number_of_edges()}\n"
        info += f"Average clustering coefficient: {nx.average_clustering(self.G):.4f}\n"
        try:
            info += f"Average shortest path length: {nx.average_shortest_path_length(self.G):.4f}\n"
        except nx.NetworkXError:
            info += "Average shortest path length: N/A (Graph is not connected)\n"
        return info
    
    def calculate_mis(self):
        """Calculate the Maximum Independent Set (MIS) of the graph."""
        if self.G is None:
            print("No graph generated yet. Please generate a graph first.")
            return

        def mis(G):
            if G.number_of_nodes() == 0:
                return set()

            # Step 1: Check if maximum degree is at most 2
            if max(dict(G.degree()).values()) <= 2:
                # For graphs with max degree 2, we can solve MIS in polynomial time
                # This is a simplification; in practice, you'd implement a specific algorithm for this case
                return set(nx.maximal_independent_set(G))

            # Step 3: Check for degree 1 vertices
            degree_one_nodes = [n for n, d in G.degree() if d == 1]
            if degree_one_nodes:
                v = degree_one_nodes[0]
                G_minus_N_v = G.copy()
                G_minus_N_v.remove_nodes_from(list(G.neighbors(v)) + [v])
                return {v} | mis(G_minus_N_v)

            # Step 5: Check if G is not connected
            if not nx.is_connected(G):
                components = list(nx.connected_components(G))
                G1 = G.subgraph(components[0])
                G_minus_G1 = G.copy()
                G_minus_G1.remove_nodes_from(components[0])
                return mis(G1) | mis(G_minus_G1)

            # Step 8: Select a vertex with maximum degree
            v = max(G.degree(), key=lambda x: x[1])[0]
            G_minus_N_v = G.copy()
            G_minus_N_v.remove_nodes_from(list(G.neighbors(v)) + [v])
            G_minus_v = G.copy()
            G_minus_v.remove_node(v)

            mis_with_v = {v} | mis(G_minus_N_v)
            mis_without_v = mis(G_minus_v)

            return mis_with_v if len(mis_with_v) > len(mis_without_v) else mis_without_v

        solution_folder = os.path.join(self.mis_folder, self.graph_type)
        if not os.path.exists(solution_folder):
            os.makedirs(solution_folder)

        # Calculate MIS
        mis_result = mis(self.G)

        # Visualize the solution
        self._visualize_mis(mis_result, solution_folder)

        return mis_result

    def _visualize_mis(self, mis_set, folder):
        """Visualize the MIS solution."""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.G)
        nx.draw_networkx_nodes(self.G, pos, node_color='lightblue', node_size=500)
        nx.draw_networkx_nodes(self.G, pos, nodelist=list(mis_set), node_color='red', node_size=500)
        nx.draw_networkx_edges(self.G, pos)
        nx.draw_networkx_labels(self.G, pos)
        plt.title(f"MIS for {self.graph_type}")
        plt.axis('off')
        plt.tight_layout()

        # Save the visualization
        plt.savefig(os.path.join(folder, f"{self.graph_type}_mis.png"))
        plt.close()

        # Create an interactive visualization
        net = Network(notebook=True, cdn_resources='in_line')
        net.from_nx(self.G)
        for node in self.G.nodes():
            if node in mis_set:
                net.nodes[node]['color'] = 'red'
            else:
                net.nodes[node]['color'] = 'lightblue'
        net.show_buttons(filter_=['physics'])
        net.save_graph(os.path.join(folder, f"{self.graph_type}_mis_interactive.html"))






class GraphGenerator_old:
    def __init__(self):
        """Initialize the GraphGenerator with default values."""
        self.G = None
        self.graph_type = ""
        self.output_folder = "graphs"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def generate_erdos_renyi(self, n, p):
        """Generate an Erdős-Rényi graph."""
        self.G = nx.erdos_renyi_graph(n, p)
        self.graph_type = f"Erdos-Renyi_n{n}_p{p}"

    def generate_barabasi_albert(self, n, m):
        """Generate a Barabási-Albert graph."""
        self.G = nx.barabasi_albert_graph(n, m)
        self.graph_type = f"Barabasi-Albert_n{n}_m{m}"

    def generate_watts_strogatz(self, n, k, p):
        """Generate a Watts-Strogatz graph."""
        self.G = nx.watts_strogatz_graph(n, k, p)
        self.graph_type = f"Watts-Strogatz_n{n}_k{k}_p{p}"

    def generate_random_regular(self, n, d):
        """Generate a Random Regular graph."""
        self.G = nx.random_regular_graph(d, n)
        self.graph_type = f"Random-Regular_n{n}_d{d}"

    def visualize_graph_interactive(self, save=True, cdn_resources='in_line'):
        """
        Visualize the generated graph using Pyvis for an interactive HTML file.
        
        Parameters:
        save (bool): Whether to save the graph as an HTML file. Default is True.
        cdn_resources (str): The source of CDN resources. Options are 'in_line', 'remote', or 'local'. Default is 'in_line'.
        """
        if self.G is None:
            print("No graph generated yet. Please generate a graph first.")
            return

        net = Network(notebook=True, cdn_resources=cdn_resources)
        net.from_nx(self.G)
        net.show_buttons(filter_=['physics'])

        html_filename = f"{self.graph_type}_plot.html"
        html_path = os.path.join(self.output_folder, html_filename)

        if save:
            net.save_graph(html_path)
            print(f"Graph plot saved as: {html_path}")

        # Display the saved HTML file content in the Jupyter notebook
        display(HTML(html_path))

    def get_graph_info(self):
        """
        Get information about the generated graph.
        
        Returns:
        str: A string containing information about the graph.
        """
        if self.G is None:
            return "No graph generated yet."

        info = f"Graph Type: {self.graph_type}\n"
        info += f"Number of nodes: {self.G.number_of_nodes()}\n"
        info += f"Number of edges: {self.G.number_of_edges()}\n"
        info += f"Average clustering coefficient: {nx.average_clustering(self.G):.4f}\n"
        try:
            info += f"Average shortest path length: {nx.average_shortest_path_length(self.G):.4f}\n"
        except nx.NetworkXError:
            info += "Average shortest path length: N/A (Graph is not connected)\n"
        return info

# # Example usage in a Jupyter notebook
# if __name__ == "__main__":
#     generator = GraphGenerator()

#     # Generate and visualize an Erdős-Rényi graph
#     generator.generate_erdos_renyi(n=20, p=0.2)
#     print(generator.get_graph_info())
#     generator.visualize_graph_interactive()

#     # Generate and visualize a Barabási-Albert graph
#     generator.generate_barabasi_albert(n=20, m=2)
#     print(generator.get_graph_info())
#     generator.visualize_graph_interactive()

#     # Generate and visualize a Watts-Strogatz graph
#     generator.generate_watts_strogatz(n=20, k=4, p=0.1)
#     print(generator.get_graph_info())
#     generator.visualize_graph_interactive()

#     # Generate and visualize a Random Regular graph
#     generator.generate_random_regular(n=20, d=3)
#     print(generator.get_graph_info())
#     generator.visualize_graph_interactive()
