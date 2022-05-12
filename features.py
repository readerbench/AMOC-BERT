from graph.graph import AmocGraph
from typing import List, Tuple
import numpy as np
import networkx as nx


class Features:
    
    def __init__(self, use_aoe: bool):
        self.use_aoe = use_aoe
        self.magic_number = 3
        
    def compute_all_metrics(self, intermidiate_graphs: List[AmocGraph]) -> List[float]:
        metrics = {}
        metrics["mean_score_for_active_nodes"] = self.compute_mean_score_for_active_nodes(intermidiate_graphs)
        metrics["mean_score_for_all_nodes"] = self.compute_mean_score_for_all_nodes(intermidiate_graphs)
        metrics["mean_delta_active_nodes_between_two_sentences"] = self.compute_mean_delta_active_nodes_between_two_sentences(intermidiate_graphs)
        
        only_active_nodes_nx_graphs, all_nodes_nx_graphs = self.build_nx_graphs_for_itermidiate_amoc_graphs(intermidiate_graphs)
        
        metrics["mean_degree_centrality_active_nodes"] = self.compute_mean_degree_centrality(only_active_nodes_nx_graphs)
        metrics["mean_degree_centrality_all_nodes"] = self.compute_mean_degree_centrality(all_nodes_nx_graphs)
        metrics["mean_closeness_centrality_active_nodes"] = self.compute_mean_closeness_centrality(only_active_nodes_nx_graphs)
        metrics["mean_closeness_centrality_all_nodes"] = self.compute_mean_closeness_centrality(all_nodes_nx_graphs)
        metrics["mean_betweenness_centrality_active_nodes"] = self.compute_mean_betweenness_centrality(only_active_nodes_nx_graphs)
        metrics["mean_betweenness_centrality_all_nodes"] = self.compute_mean_betweenness_centrality(all_nodes_nx_graphs)
        metrics["mean_harmonic_centrality_active_nodes"] = self.compute_mean_harmonic_centrality(only_active_nodes_nx_graphs)
        metrics["mean_harmonic_centrality_all_nodes"] = self.compute_mean_harmonic_centrality(all_nodes_nx_graphs)
        metrics["mean_density_active_nodes"] = self.compute_mean_density(only_active_nodes_nx_graphs)
        metrics["mean_density_all_nodes"] = self.compute_mean_density(all_nodes_nx_graphs)
        metrics["mean_modularity_active_nodes"] = self.compute_mean_modularity(only_active_nodes_nx_graphs)
        metrics["mean_modularity_all_nodes"] = self.compute_mean_modularity(all_nodes_nx_graphs)
        
        return metrics
    
    def compute_mean_score_for_active_nodes(self, intermidiate_graphs: List[AmocGraph]) -> float:
        scores = []
        for graph in intermidiate_graphs:
            nodes_scores = graph.get_max_active_nodes_by_score(use_aoe=self.use_aoe)
            scores.append(np.mean(list(nodes_scores.values())))
        return np.mean(scores)
    
    def compute_mean_score_for_all_nodes(self, intermidiate_graphs: List[AmocGraph]) -> float:
        scores = []
        for graph in intermidiate_graphs:
            nodes_scores = graph.get_top_n_nodes_by_score(top_n=len(graph.nodes),use_aoe=self.use_aoe)
            scores.append(np.mean(list(nodes_scores.values())))
        return np.mean(scores)
    
    def compute_mean_delta_active_nodes_between_two_sentences(self, intermidiate_graphs: List[AmocGraph]) -> float:
        scores = []
        for i in range(len(intermidiate_graphs) - 1):
            graph1 = intermidiate_graphs[i]
            graph2 = intermidiate_graphs[i + 1]
            active_nodes_graph1 = set(graph1.get_max_active_nodes_by_score(use_aoe=self.use_aoe).keys())
            active_nodes_graph2 = set(graph2.get_max_active_nodes_by_score(use_aoe=self.use_aoe).keys())
            scores.append(len(active_nodes_graph1.intersection(active_nodes_graph2)))
        return np.mean(scores)
    
    def build_nx_graph(self, graph: AmocGraph, only_active: bool) -> nx.Graph:
        nx_graph = nx.Graph()
        for node in graph.nodes:
            if only_active and not node.active:
                continue
            nx_graph.add_node(node.text)
        for node in graph.nodes:
            if only_active and not node.active:
                continue
            # if only_active:
            #     print(node.text, graph.edges[node])
            for edge in graph.edges[node]:
                if only_active and not edge[0].active:
                    continue
                nx_graph.add_edge(node.text, edge[0].text, weigth = self.magic_number - edge[1])
        return nx_graph
    
    def build_nx_graphs_for_itermidiate_amoc_graphs(self, intermidiate_graphs: List[AmocGraph]) -> Tuple[List[nx.Graph], List[nx.Graph]]:
        active_graphs = []
        inactive_graphs = []
        for graph in intermidiate_graphs:
            active_graphs.append(self.build_nx_graph(graph, only_active=True))
            inactive_graphs.append(self.build_nx_graph(graph, only_active=False))
        return active_graphs, inactive_graphs
    
    def compute_mean_degree_centrality(self, intermidiate_graphs: List[nx.Graph]) -> float:
        scores = []
        for graph in intermidiate_graphs:
            scores.append(np.mean(list(nx.algorithms.centrality.degree_centrality(graph).values())))
        return np.mean(scores)
    
    def compute_mean_closeness_centrality(self, intermidiate_graphs: List[nx.Graph]) -> float:
        scores = []
        for graph in intermidiate_graphs:
            scores.append(np.mean(list(nx.algorithms.centrality.closeness_centrality(graph).values())))
        return np.mean(scores)
    
    def compute_mean_betweenness_centrality(self, intermidiate_graphs: List[nx.Graph]) -> float:
        scores = []
        for graph in intermidiate_graphs:
            scores.append(np.mean(list(nx.algorithms.centrality.betweenness_centrality(graph).values())))
        return np.mean(scores)
    
    def compute_mean_harmonic_centrality(self, intermidiate_graphs: List[nx.Graph]) -> float:
        scores = []
        for graph in intermidiate_graphs:
            scores.append(np.mean(list(nx.algorithms.centrality.harmonic_centrality(graph).values())))
        return np.mean(scores)
    
    def compute_mean_density(self, intermidiate_graphs: List[nx.Graph]) -> float:
        scores = []
        for graph in intermidiate_graphs:
            scores.append(nx.classes.function.density(graph))
        return np.mean(scores)
    
    def compute_mean_modularity(self, intermidiate_graphs: List[nx.Graph]) -> float:
        scores = []
        for graph in intermidiate_graphs:
            if len(graph.nodes) <= 1:
                scores.append(0)
            else:
                try:
                    scores.append(len(list(nx.algorithms.community.modularity_max.greedy_modularity_communities(graph))))
                except:
                    scores.append(0)
        return np.mean(scores)
    

if __name__ == "__main__":
    intermidiate_graphs = []
    for i in range(13):
        intermidiate_graphs.append(AmocGraph.load_graph_from_pickle(i))
    features = Features(use_aoe=True)
    metrics = features.compute_all_metrics(intermidiate_graphs)
    print(metrics)