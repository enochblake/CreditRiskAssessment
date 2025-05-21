import matplotlib.pyplot as plt
import networkx as nx
from inference import build_model  # Make sure this returns your learned Bayesian Network

def plot_bn():
    model = build_model()
    edges = model.edges()
    
    G = nx.DiGraph()
    G.add_edges_from(edges)

    # Use spring layout for visualization
    pos = nx.spring_layout(G, seed=42)  # seed for reproducibility

    # Draw nodes, edges, and labels
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color="lightblue")
    nx.draw_networkx_edges(G, pos, arrowsize=20, arrowstyle='-|>')
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

    plt.title("Credit Risk Bayesian Network Structure")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_bn()
