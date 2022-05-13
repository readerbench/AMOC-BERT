from typing import List, Dict, Tuple, Set
from graph.node import AmocNode, AmocNodeType
import networkx as nx
import pylab as plt
import pickle
import csv
import math
from spacy.tokens import Doc, Token, Span


class AmocGraph:
    
    def __init__(self, max_active_concepts=3):
        self.nodes: Set[AmocNode] = set()
        self.edges: Dict[AmocNode, List[Tuple[AmocNode, float]]] = {}
        self.max_active_concepts = max_active_concepts
        self.aoe_dict = self.load_aoe_dict()
        self.total_number_of_attentions = 0
    
    def load_aoe_dict(self):
        aoe_dict = {}
        with open("aoe/aoe.csv", "r") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                aoe_dict[row[1]] = math.log10(float(row[0]))
        return aoe_dict
        
    def add_node(self, node: AmocNode):
        if node not in self.nodes:
            self.nodes.add(node)
            self.edges[node] = []
        else:
            for graph_node in self.nodes:
                if graph_node == node:
                    if graph_node.node_type == AmocNodeType.INFERRED and node.node_type == AmocNodeType.TEXT_BASED:
                        graph_node.node_type = AmocNodeType.TEXT_BASED
                    break
    
    def get_graph_node_from_node(self, node: AmocNode) -> AmocNode:
        if node not in self.nodes:
            return None
        for local_node in self.nodes:
            if local_node == node:
                return local_node
        return None
    
    def get_graph_node_from_text(self, text: str) -> AmocNode:
        for node in self.nodes:
            if node.text == text:
                return node
        return None
    
    def add_edge(self, node: AmocNode, edge: Tuple[AmocNode, float]):
        self.edges[node].append(edge)
        
    def add_edge_both_ways(self, node: AmocNode, edge: Tuple[AmocNode, float]):
        if node in self.nodes and edge[0] in self.nodes: # this will NOT happen when there is a spacy error -> ex. lemma of born is bear
            self.add_edge(node, edge)
            self.add_edge(edge[0], (node, edge[1]))
        
    def get_nodes(self) -> Set[AmocNode]:
        return self.nodes
    
    def get_edges(self) -> Dict[AmocNode, List[Tuple[AmocNode, float]]]:
        return self.edges
    
    def get_node_score(self, node: AmocNode) -> float:
        return sum(edge[1] for edge in self.edges[node]) / self.total_number_of_attentions
    
    def get_scale_factor_by_aoe(self, use_aoe:bool, node: AmocNode) -> float:
        if not use_aoe:
            return 1.0
        return self.aoe_dict.get(node.text, 1.0)
    
    def get_top_n_nodes_by_score(self, top_n, use_aoe=False) -> List[Tuple[AmocNode, float]]:
        return {word: self.get_node_score(word) * self.get_scale_factor_by_aoe(use_aoe, word) for word, neighbours in dict(sorted(self.edges.items(), key=lambda x: self.get_node_score(x[0]) * self.get_scale_factor_by_aoe(use_aoe, x[0]), reverse=True)[:top_n]).items()}

    def get_max_active_nodes_by_score(self, only_text_based:bool=False, use_aoe:bool=False) -> List[Tuple[AmocNode, float]]:
        sorted_edges = sorted(self.edges.items(), key=lambda x: self.get_node_score(x[0]) * self.get_scale_factor_by_aoe(use_aoe, x[0]), reverse=True)
        return {word: self.get_node_score(word) * self.get_scale_factor_by_aoe(use_aoe, word) for word, neighbours in dict(sorted_edges[:self.max_active_concepts]).items() if word.node_type == AmocNodeType.TEXT_BASED or not only_text_based}
    
    def activate_and_deactivate_nodes(self, use_aoe=False) -> AmocNode:
        word_to_score = self.get_max_active_nodes_by_score()
        for word in word_to_score:
            word.active = True
        for word in self.nodes:
            if word not in word_to_score:
                word.active = False
                
    def decay_edges_with_percentage(self, percentage) -> None:
        for node, neighbours in self.edges.items():
            updated_neighbours = []
            for neighbour in neighbours:
                updated_value = neighbour[1] - neighbour[1] * percentage
                updated_neighbours.append((neighbour[0], updated_value))
            self.edges[node] = updated_neighbours
    
    def draw_graph_top_n(self, iteration, top_n=15, use_aoe=False):
        tmp = self.max_active_concepts
        self.max_active_concepts = top_n
        top_nodes = set(self.get_max_active_nodes_by_score(use_aoe=use_aoe).keys())
        self.max_active_concepts = tmp
        G = nx.Graph()
        color_map = []
        for node in top_nodes:
            G.add_node(node.text)
            color = None
            if node.node_type == AmocNodeType.INFERRED:
                if node.active:
                    color = "#D22B2B"
                else:
                    color = "#FA8072"
            else:
                if node.active:
                    color = "#0047AB"
                else:
                    color = "#ADD8E6"
            color_map.append(color)
        for node, neighbours in self.edges.items():
            if node in top_nodes:
                for neighbour in neighbours:
                    if neighbour[0] in top_nodes:
                        if not G.has_edge(node.text, neighbour[0].text):
                            G.add_edge(node.text, neighbour[0].text, weight=neighbour[1])
        nx.draw(G, with_labels=True, node_color=color_map)
        plt.savefig(f"viz/{iteration}.png")
        plt.close()
        
    
    def draw_full_graph(self, iteration):
        G = nx.Graph()
        color_map = []
        for node in self.nodes:
            G.add_node(node.text)
            color = None
            if node.node_type == AmocNodeType.INFERRED:
                if node.active:
                    color = "#D22B2B"
                else:
                    color = "#FA8072"
            else:
                if node.active:
                    color = "#0047AB"
                else:
                    color = "#ADD8E6"
            color_map.append(color)
        for node, neighbours in self.edges.items():
            for neighbour in neighbours:
                if not G.has_edge(node.text, neighbour[0].text):
                    G.add_edge(node.text, neighbour[0].text, weight=neighbour[1])
        nx.draw(G, with_labels=True, node_color=color_map)
        plt.savefig(f"viz/{iteration}.png")
        plt.close()
        
    
    def save_graph_to_pickle(self, iteration, folder_path=None):
        if not folder_path:
            with open(f"graph_pickle_save/graph_{iteration}.pickle", "wb") as f:
                pickle.dump(self, f)
        else:
            with open(f"{folder_path}/graph_{iteration}.pickle", "wb") as f:
                pickle.dump(self, f)
    
    @staticmethod    
    def load_graph_from_pickle(iteration, folder_path=None):
        if not folder_path:
            with open(f"graph_pickle_save/graph_{iteration}.pickle", "rb") as f:
                return pickle.load(f)
        else:
            with open(f"{folder_path}/graph_{iteration}.pickle", "rb") as f:
                return pickle.load(f)
    
            
    def __str__(self) -> str:
        return str(self.edges)
    
    def __repr__(self) -> str:
        return str(self.edges)
        