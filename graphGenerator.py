import networkx as nx
import os
from pyvis.network import Network
from IPython.display import display, HTML

class GraphGenerator:
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

    def visualize_graph_interactive(self, title, save=True, cdn_resources='in_line'):
        """
        Visualize the generated graph using Pyvis for an interactive HTML file.
        
        Parameters:
        title (str): The title of the graph.
        save (bool): Whether to save the graph as an HTML file. Default is True.
        cdn_resources (str): The source of CDN resources. Options are 'in_line', 'remote', or 'local'. Default is 'in_line'.
        """
        if self.G is None:
            print("No graph generated yet. Please generate a graph first.")
            return

        net = Network(notebook=True, cdn_resources=cdn_resources)
        net.from_nx(self.G)
        net.show_buttons(filter_=['physics'])

        if save:
            html_path = os.path.join(self.output_folder, f"{self.graph_type}_plot.html")
            net.save_graph(html_path)
            print(f"Graph plot saved as: {html_path}")

        net.show(title)

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
#     generator.visualize_graph_interactive("Erdős-Rényi Random Graph")

#     # Generate and visualize a Barabási-Albert graph
#     generator.generate_barabasi_albert(n=20, m=2)
#     print(generator.get_graph_info())
#     generator.visualize_graph_interactive("Barabási-Albert Preferential Attachment Graph")

#     # Generate and visualize a Watts-Strogatz graph
#     generator.generate_watts_strogatz(n=20, k=4, p=0.1)
#     print(generator.get_graph_info())
#     generator.visualize_graph_interactive("Watts-Strogatz Small-World Graph")

#     # Generate and visualize a Random Regular graph
#     generator.generate_random_regular(n=20, d=3)
#     print(generator.get_graph_info())
#     generator.visualize_graph_interactive("Random Regular Graph")
