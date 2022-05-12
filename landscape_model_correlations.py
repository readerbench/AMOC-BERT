import json
import pickle
from graph.graph import AmocGraph
from pprint import pprint
from scipy.stats import spearmanr, pearsonr
import numpy as np


def load_lm_json():
    with open("landscape_model/lm_activation_scores.json", "r") as f:
        return json.load(f)
    

def compute_Pearson_correlation(lm_scores, amoc_activation_scores, consider_inferred=True):
    pearson_correlations = {}
    for word in lm_scores:
        if not consider_inferred and lm_scores[word]["type"] == "INFERRED":
            continue
        lm_word_scores = lm_scores[word]["values"]
        amoc_word_scores = amoc_activation_scores[word]
        pearson_correlations[word] = pearsonr(lm_word_scores, amoc_word_scores)[0]
    return pearson_correlations


def compute_Spearman_correlation(lm_scores, amoc_activation_scores, consider_inferred=True):
    spearman_correlations = {}
    for word in lm_scores:
        if not consider_inferred and lm_scores[word]["type"] == "INFERRED":
            continue
        lm_word_scores = lm_scores[word]["values"]
        amoc_word_scores = amoc_activation_scores[word]
        spearman_correlations[word] = spearmanr(lm_word_scores, amoc_word_scores).correlation
    return spearman_correlations


def compute_Pearson_correlation_per_sentence(lm_scores, amoc_activation_scores, consider_inferred=True):
    pearson_correlations = {}
    for sentence_index in range(13):
        lm_sentence_scores = []
        amoc_sentence_scores = []
        for word in lm_scores:
            if not consider_inferred and lm_scores[word]["type"] == "INFERRED":
                continue
            lm_sentence_scores.append(lm_scores[word]["values"][sentence_index])
            amoc_sentence_scores.append(amoc_activation_scores[word][sentence_index])
        pearson_correlations[sentence_index + 1] = pearsonr(lm_sentence_scores, amoc_sentence_scores)[0]
    return pearson_correlations


def compute_Spearman_correlation_per_sentence(lm_scores, amoc_activation_scores, consider_inferred=True):
    spearman_correlations = {}
    for sentence_index in range(13):
        lm_sentence_scores = []
        amoc_sentence_scores = []
        for word in lm_scores:
            if not consider_inferred and lm_scores[word]["type"] == "INFERRED":
                continue
            lm_sentence_scores.append(lm_scores[word]["values"][sentence_index])
            amoc_sentence_scores.append(amoc_activation_scores[word][sentence_index])
        spearman_correlations[sentence_index + 1] = spearmanr(lm_sentence_scores, amoc_sentence_scores).correlation
    return spearman_correlations


if __name__ == "__main__":
    lm_scores = load_lm_json()
    words = list(lm_scores.keys())
    amoc_activation_scores = {w: [] for w in words}
    for i in range(13):
        graph = AmocGraph.load_graph_from_pickle(i)
        for word in words:
            word_node = graph.get_graph_node_from_text(word)
            if not word_node:
                amoc_activation_scores[word].append(0)
                continue
            amoc_activation_scores[word].append(graph.get_node_score(word_node))
    # spearman_correlations = compute_Spearman_correlation(lm_scores, amoc_activation_scores, consider_inferred=False)
    # pprint(spearman_correlations)
    # sp_values = np.nan_to_num(list(spearman_correlations.values()), nan=0)
    # print(f"Mean: {np.mean(sp_values)}, Std: {np.std(sp_values)}")
    
    # pearson_correlations = compute_Pearson_correlation(lm_scores, amoc_activation_scores, consider_inferred=True)
    # pprint(pearson_correlations)
    # sp_values = np.nan_to_num(list(pearson_correlations.values()), nan=0)
    # print(f"Mean: {np.mean(sp_values)}, Std: {np.std(sp_values)}")
    
    # pearson_correlations = compute_Pearson_correlation(lm_scores, amoc_activation_scores, consider_inferred=False)
    # pprint(pearson_correlations)
    # sp_values = np.nan_to_num(list(pearson_correlations.values()), nan=0)
    # print(f"Mean: {np.mean(sp_values)}, Std: {np.std(sp_values)}")
    
    pearson_correlations = compute_Spearman_correlation_per_sentence(lm_scores, amoc_activation_scores, consider_inferred=False)
    pprint(pearson_correlations)
    sp_values = np.nan_to_num(list(pearson_correlations.values()), nan=0)
    print(f"Mean: {np.mean(sp_values)}, Std: {np.std(sp_values)}")
    
    pearson_correlations = compute_Pearson_correlation_per_sentence(lm_scores, amoc_activation_scores, consider_inferred=False)
    pprint(pearson_correlations)
    sp_values = np.nan_to_num(list(pearson_correlations.values()), nan=0)
    print(f"Mean: {np.mean(sp_values)}, Std: {np.std(sp_values)}")