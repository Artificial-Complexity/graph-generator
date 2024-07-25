import networkx as nx
import matplotlib.pyplot as plt
import random
from pyvis.network import Network
import os
from IPython.display import HTML

class GraphGenerator:
    def __init__(self):
        self.G = None
        self.graph_type = ""
        self.output_folder = "graphs"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def generate_erdos_renyi(self, n, p):
        self.G = nx.erdos_renyi_graph(n, p)
        self.graph_type = f"Erdos-Renyi_n{n}_p{p}"

    def generate_barabasi_albert(self, n, m):
        self.G = nx.barabasi_albert_graph(n, m)
        self.graph_type = f"Barabasi-Albert_n{n}_m{m}"

    def generate_watts_strogatz(self, n, k, p):
        self.G = nx.watts_strogatz_graph(n, k, p)
        self.graph_type = f"Watts-Strogatz_n{n}_k{k}_p{p}"

    def generate_random_regular(self, n, d):
        self.G = nx.random_regular_graph(d, n)
        self.graph_type = f"Random-Regular_n{n}_d{d}"

    def visualize_graph(self, title):
        if self.G is None:
            print("No graph generated yet. Please generate a graph first.")
            return

        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(self.G)
        nx.draw(self.G, pos, with_labels=True, node_color='lightblue', 
                node_size=500, font_size=10, font_weight='bold')
        plt.title(title)
        plt.axis('off')
        
        img_path = os.path.join(self.output_folder, f"{self.graph_type}_plot.png")
        plt.savefig(img_path)
        plt.close()
        print(f"Graph plot saved as: {img_path}")

    def get_graph_html(self):
        if self.G is None:
            print("No graph generated yet. Please generate a graph first.")
            return

        nt = Network(notebook=True, height="500px", width="100%", bgcolor="#222222", font_color="white")
        nt.from_nx(self.G)
        nt.toggle_physics(True)
        return nt.generate_html()

    def display_graph_html(self):
        """
        Display the graph as an interactive HTML visualization in the Jupyter notebook.
        """
        html_content = self.get_graph_html()
        return HTML(html_content)

    def get_graph_info(self):
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
#     generator.visualize_graph("Erdős-Rényi Random Graph")
#     generator.display_graph_html()

#     # Generate and visualize a Barabási-Albert graph
#     generator.generate_barabasi_albert(n=20, m=2)
#     print(generator.get_graph_info())
#     generator.visualize_graph("Barabási-Albert Preferential Attachment Graph")
#     generator.display_graph_html()

#     # Generate and visualize a Watts-Strogatz graph
#     generator.generate_watts_strogatz(n=20, k=4, p=0.1)
#     print(generator.get_graph_info())
#     generator.visualize_graph("Watts-Strogatz Small-World Graph")
#     generator.display_graph_html()

#     # Generate and visualize a Random Regular graph
#     generator.generate_random_regular(n=20, d=3)
#     print(generator.get_graph_info())
#     generator.visualize_graph("Random Regular Graph")
#     generator.display_graph_html()