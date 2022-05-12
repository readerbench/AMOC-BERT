import spacy
from spacy.tokens import Doc, Token, Span
from spacy.lang.en import English
from typing import List, Dict, Tuple
from transformers import DistilBertTokenizer, TFDistilBertModel, DistilBertConfig
from transformers import GPT2Tokenizer, GPT2Config, TFGPT2Model
from transformers import pipeline, set_seed
from graph.graph import AmocGraph
from graph.node import AmocNode, AmocNodeType
import re
import numpy as np


class AMOCBert():
    
    def __init__(self, bert_model_name: str, spacy_model_name: str, generator_model_name: str, 
                    number_of_sentences_lookback: int, maximum_active_concepts: int,
                    maximum_dictionary_expansion: int, attention_score_threshold: float, generation_imagination_factor: int,
                    weigth_decay_percatage: float, use_aoe: bool, folder_to_save_graphs: str = None):
        self.init_bert_model(bert_model_name)
        self.number_of_sentences_lookback = number_of_sentences_lookback
        self.maximum_active_concepts = maximum_active_concepts
        self.maximum_dictionary_expansion = maximum_dictionary_expansion
        self.attention_score_threshold = attention_score_threshold
        self.spacy_model = spacy.load(spacy_model_name)
        self.amoc_graph = AmocGraph(max_active_concepts=self.maximum_active_concepts)
        self.generator = pipeline('text-generation', model=generator_model_name)
        self.set_top_k_and_top_p_based_on_imagination(generation_imagination_factor)
        self.weigth_decay_percatage = weigth_decay_percatage
        self.use_aoe = use_aoe
        self.folder_to_save_graphs = folder_to_save_graphs
        set_seed(42)
        
    
    def set_top_k_and_top_p_based_on_imagination(self, generation_imagination_factor):
        if generation_imagination_factor == 0: # smallest imagination and the generation will not vary that much
            self.top_k = 30
            self.top_p = 0.7
        elif generation_imagination_factor == 1:
            self.top_k = 40
            self.top_p = 0.8
        elif generation_imagination_factor == 2:
            self.top_k = 50
            self.top_p = 0.9
        elif generation_imagination_factor == 3:
            self.top_k = 60
            self.top_p = 0.95
        else:
            self.top_k = 0
            self.top_p = 0
    
    
    def init_bert_model(self, bert_model_name: str) -> None:
        if bert_model_name == 'distilbert-base-uncased':
            self.bert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            self.bert_config = DistilBertConfig.from_pretrained('distilbert-base-uncased', output_hidden_states=True, output_attentions=True)
            self.bert_model = TFDistilBertModel.from_pretrained("distilbert-base-uncased", config=self.bert_config)
        elif bert_model_name == 'gpt2':
            self.bert_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.bert_config = GPT2Config.from_pretrained('gpt2', output_hidden_states=True, output_attentions=True)
            self.bert_model = TFGPT2Model.from_pretrained("gpt2", config=self.bert_config)
        else:
            print("Error! The bert model not found or not supported!")
            self.bert_tokenizer = None
            self.bert_config = None
            self.bert_model = None
    
    def clean_text(self, text: str) -> str:
        return text.replace("’", "'") \
            .replace("“", '"') \
            .replace("”", '"') \
            .replace("「", '') \
            .replace("」", '') \


    def common_prefix(self, word1: str, word2: str) -> str:
        i = 0
        while i < min(len(word1), len(word2)):
            if word1[i] != word2[i]:
                return word1[:i]
            i += 1
        return word1[:i]


    def create_word_token_dict(self, spacy_doc: Doc, tokenized_sequence: List[str]) -> Dict[Token, List[int]]:
        i = 0
        result = {}
        words = [self.clean_text(word.text) for word in spacy_doc]
        if getattr(self.bert_tokenizer, "do_lower_case", False):
            words = [word.lower() for word in words]
        block_symbols = {s for word in words for s in word}
        tokens = ["".join(s for s in token if s in block_symbols) for token in tokenized_sequence]
        current = ""
        for text, word in zip(words, spacy_doc):
            if i >= len(tokens):
                break
            while not tokens[i]:
                print(i, tokens[i])
                i += 1
                if i >= len(tokens):
                    break
            if i >= len(tokens):
                break
            current = self.common_prefix(text, tokens[i])
            if not current:
                continue 
            ids = [i]
            if text == current:
                if len(text) < len(tokens[i]):
                    tokens[i] = tokens[i][len(current):]
                else:
                    i += 1
            else: 
                i += 1
                while i < len(tokens) and text.startswith(current + tokens[i]):
                    ids.append(i)
                    current += tokens[i]
                    i += 1
                current = text[len(current):]
                if len(current) > 0:
                    tokens[i] = tokens[i][len(current):]
            result[word] = ids
        return result


    def is_content_word_and_non_stopword(self, token: Token, including_pronouns: bool = False) -> bool:
        pos_list = ["NOUN", "PROPN", "ADJ", "VERB", "ADV"]
        if including_pronouns:
            pos_list.append("PRON")
        return (token.pos_ in pos_list) and (token.lemma_ not in self.spacy_model.Defaults.stop_words)


    def analyze_only_one_sentence(self, text: str) -> None:
        spacy_doc = self.spacy_model(text)
        tokenized_sequence_str = self.bert_tokenizer.tokenize(text)
        word_to_tokens_dict = self.create_word_token_dict(spacy_doc, tokenized_sequence_str)
        last_sentence = list(spacy_doc.sents)[-1]
        content_words_from_last_sentence = [word for word in last_sentence if self.is_content_word_and_non_stopword(word)]
        tokenized_sequence_tensor = self.bert_tokenizer(text, return_tensors='tf')
        outputs = self.bert_model(tokenized_sequence_tensor)
        attentions = outputs[-1]
        for word in content_words_from_last_sentence:
            node = AmocNode(word.lemma_, AmocNodeType.TEXT_BASED)
            self.amoc_graph.add_node(node)
        for i, word1 in enumerate(content_words_from_last_sentence):
            for j, word2 in enumerate(content_words_from_last_sentence):
                if i == j:
                    continue
                score = 0.0
                for token_id in word_to_tokens_dict[word1]:
                    for token_id2 in word_to_tokens_dict[word2]:
                        for head in attentions:
                            for layer in head[0]:
                                score += layer[token_id][token_id2].numpy()
                if score > 0.0:
                    # node1 = AmocNode(word1.lemma_, AmocNodeType.TEXT_BASED)
                    # node2 = AmocNode(word2.lemma_, AmocNodeType.TEXT_BASED)
                    node1 = self.amoc_graph.get_graph_node_from_text(word1.lemma_)
                    node2 = self.amoc_graph.get_graph_node_from_text(word2.lemma_)
                    self.amoc_graph.add_edge_both_ways(node1, (node2, score))
        self.amoc_graph.total_number_of_attentions += len(attentions) * len(attentions[0][0])
                    

    def analyze_last_sentences(self, text: str, text_components_char_start: Dict, number_of_last_sentences: int, generation=False) -> Tuple[Dict, Dict]:
        print(text)
        spacy_doc = self.spacy_model(text)
        tokenized_sequence_str = self.bert_tokenizer.tokenize(text)
        word_to_tokens_dict = self.create_word_token_dict(spacy_doc, tokenized_sequence_str)
        last_sentences = list(spacy_doc.sents)[-number_of_last_sentences:]
        content_words_from_last_sentences = [word for sentence in last_sentences for word in sentence if self.is_content_word_and_non_stopword(word)]
        content_words_from_last_sentences = self.filter_out_text_based_words_from_generation(number_of_last_sentences, generation, spacy_doc, content_words_from_last_sentences)
        print(f"Content words for last sentence: {content_words_from_last_sentences}")
        # start_index_of_generation = last_sentences[0][0].idx
        start_index_of_generation = text_components_char_start['current_start']
        tokenized_sequence_tensor = self.bert_tokenizer(text, return_tensors='tf')
        outputs = self.bert_model(tokenized_sequence_tensor)
        attentions = outputs[-1]
        
        for word in content_words_from_last_sentences:
            node_type = AmocNodeType.INFERRED if generation else AmocNodeType.TEXT_BASED
            node = AmocNode(word.lemma_, node_type)
            self.amoc_graph.add_node(node)
        all_content_words = [word for word in spacy_doc if self.is_content_word_and_non_stopword(word)]
        words_scores = {}
        words_maximum_scores = {}
        for word1 in content_words_from_last_sentences:
            words_scores[word1] = {}
            words_maximum_scores[word1] = 0.0
            for word2 in all_content_words:
                if word1.idx == word2.idx:
                    continue
                score = 0.0
                max_attention_scores_per_token_ids = {}
                for token_id in word_to_tokens_dict[word1]:
                    for token_id2 in word_to_tokens_dict[word2]:
                        for head in attentions:
                            for layer in head[0]:
                                attention_score = layer[token_id][token_id2].numpy()
                                score += attention_score
                                if token_id not in max_attention_scores_per_token_ids:
                                    max_attention_scores_per_token_ids[token_id] = attention_score
                                else:
                                    if attention_score > max_attention_scores_per_token_ids[token_id]:
                                        max_attention_scores_per_token_ids[token_id] = attention_score
                words_scores[word1][word2] = score * self.get_word2_factor(word2, text_components_char_start)
                words_maximum_scores[word1] = max(words_maximum_scores[word1], np.mean(list(max_attention_scores_per_token_ids.values())))
        
        if not generation:
            for word1 in words_scores:
                for word2 in words_scores[word1]:
                    if words_scores[word1][word2] > 0.0:
                        # node1_type = AmocNodeType.INFERRED if generation and word1.idx >= start_index_of_generation else AmocNodeType.TEXT_BASED
                        # node2_type = AmocNodeType.INFERRED if generation and word2.idx >= start_index_of_generation else AmocNodeType.TEXT_BASED
                        # node1 = AmocNode(word1.lemma_, node1_type)
                        # node2 = AmocNode(word2.lemma_, node2_type)
                        node1 = self.amoc_graph.get_graph_node_from_text(word1.lemma_)
                        node2 = self.amoc_graph.get_graph_node_from_text(word2.lemma_)
                        self.amoc_graph.add_edge_both_ways(node1, (node2, words_scores[word1][word2]))
            self.amoc_graph.total_number_of_attentions += len(attentions) * len(attentions[0][0])
            return None, None
        else:
            self.amoc_graph.total_number_of_attentions += len(attentions) * len(attentions[0][0])
            combined_words_scores, combined_words_maximum_scores = self.combine_word_scores_based_on_lemma(words_scores, words_maximum_scores)
            return combined_words_scores, combined_words_maximum_scores
            # words_greater_than_threshold_max_scores = [word for word in combined_words_maximum_scores if combined_words_maximum_scores[word] > self.attention_score_threshold]
            # word_to_overall_score = {}
            # for word in words_greater_than_threshold_max_scores:
            #     word_to_overall_score[word] = np.sum(list(combined_words_scores[word].values()))
            # selected_words_after_cutoff = sorted(word_to_overall_score, key=word_to_overall_score.get, reverse=True)[:self.maximum_dictionary_expansion]
            # print(f"Selected words after cutoff: {selected_words_after_cutoff}")
            # for word1 in selected_words_after_cutoff:
            #     for word2 in combined_words_scores[word1]:
            #         if combined_words_scores[word1][word2] > 0.0:
            #             node1_type = AmocNodeType.INFERRED if generation and word1.idx >= start_index_of_generation else AmocNodeType.TEXT_BASED
            #             node2_type = AmocNodeType.INFERRED if generation and word2.idx >= start_index_of_generation else AmocNodeType.TEXT_BASED
            #             node1 = AmocNode(word1.lemma_, node1_type)
            #             node2 = AmocNode(word2.lemma_, node2_type)
            #             self.amoc_graph.add_edge_both_ways(node1, (node2, combined_words_scores[word1][word2]))
        
    
    def get_word2_factor(self, word2: Token, text_components_char_start: Dict) -> float:
        if word2.idx >= text_components_char_start['current_start'] - 1: # in current sentence
            return 1.0
        if word2.idx < text_components_char_start['lookback_sents_starts'][0] - 1: # in graph
            return 1.0
        factor_numitor = 2
        for start_char in text_components_char_start['lookback_sents_starts'][::-1]: # in lookback sentences
            if word2.idx >= start_char - 1:
                return 1 / factor_numitor
            else:
                factor_numitor += 1
        return 1.0
    
    
    def combine_word_scores_based_on_lemma(self, words_scores: Dict[Token, Dict[Token, float]],
                                           word_maximum_scores: Dict[Token, float]) -> Tuple[Dict[Token, Dict[Token, float]], Dict[Token, float]]:
        words_with_same_lemma = {}
        lemma_representative = {}
        for word in words_scores:
            if word.lemma_ not in lemma_representative:
                lemma_representative[word.lemma_] = word
                words_with_same_lemma[word.lemma_] = [word]
            else:
                words_with_same_lemma[word.lemma_].append(word)
        
        combined_words_scores = {}
        combined_words_maximum_scores = {}
        for word_lemma, first_word in lemma_representative.items():
            combined_words_scores[first_word] = {}
            combined_words_maximum_scores[first_word] = 0.0
            for the_word in words_with_same_lemma[word_lemma]:
                for word2 in words_scores[the_word]:
                    combined_words_scores[first_word][word2] = combined_words_scores[first_word].get(word2, 0.0) + words_scores[the_word][word2]
                combined_words_maximum_scores[first_word] = max(combined_words_maximum_scores[first_word], word_maximum_scores[the_word])
        
        return combined_words_scores, combined_words_maximum_scores
                    
                    

    def filter_out_text_based_words_from_generation(self, number_of_last_sentences: int, 
                                                    generation: bool, spacy_doc: Doc, content_words_from_last_sentences: List[Token]) -> List[Token]:
        if generation:
            filtered_content_words_from_last_sentence = []
            for word in content_words_from_last_sentences:
                found = False
                for sentence in list(spacy_doc.sents)[:-number_of_last_sentences]:
                    for text_word in sentence:
                        if word.lemma_ == text_word.lemma_:
                            found = True
                            break
                    if found:
                        break
                if not found:
                    filtered_content_words_from_last_sentence.append(word)
            content_words_from_last_sentences = filtered_content_words_from_last_sentence
        return content_words_from_last_sentences
                    

    def generate_text_and_add_to_amoc_graph(self, text: str, text_components_char_start: Dict) -> None:
        spacy_doc = self.spacy_model(text)
        max_length = len(spacy_doc) + 30
        results = self.generator(text, max_length=max_length, num_return_sequences=5, no_repeat_ngram_size=2,
                            return_full_text=False, early_stopping=True, do_sample=True, top_k=self.top_k, top_p=self.top_p)
        lemma_maximum_scores = {}
        lemma_overall_scores = {}
        lemma_to_lemma_scores = {}
        for result in results:
            generated_text = result["generated_text"].replace("\n", " ")
            generated_text = re.sub(' +', ' ', generated_text.strip())
            generated_text = self.clean_text(generated_text)
            spacy_doc_generated = self.spacy_model(generated_text)
            first_sentence = str(list(spacy_doc_generated.sents)[0])
            number_of_senteces = 1
            word_scores, word_maximum_scores = self.analyze_last_sentences(text + " " + first_sentence, text_components_char_start, number_of_senteces, generation=True)
            word_overall_scores = {}
            for word in word_scores:
                word_overall_scores[word] = np.sum(list(word_scores[word].values()))
            for word in word_overall_scores:
                if word.lemma_ not in lemma_maximum_scores:
                    lemma_maximum_scores[word.lemma_] = word_maximum_scores[word]
                    lemma_overall_scores[word.lemma_] = word_overall_scores[word]
                    lemma_to_lemma_scores[word.lemma_] = {}
                    for word2 in word_scores[word]:
                        lemma_to_lemma_scores[word.lemma_][word2.lemma_] = word_scores[word][word2]
                else:
                    lemma_maximum_scores[word.lemma_] = max(lemma_maximum_scores[word.lemma_], word_maximum_scores[word])
                    lemma_overall_scores[word.lemma_] = lemma_overall_scores[word.lemma_] + word_overall_scores[word]
                    for word2 in word_scores[word]:
                        if word2.lemma_ in lemma_to_lemma_scores[word.lemma_]:
                            lemma_to_lemma_scores[word.lemma_][word2.lemma_] = lemma_to_lemma_scores[word.lemma_][word2.lemma_] + word_scores[word][word2]
                        else:
                            lemma_to_lemma_scores[word.lemma_][word2.lemma_] = word_scores[word][word2]
                
        lemma_greater_than_threshold_max_scores = [lemma for lemma in lemma_maximum_scores if lemma_maximum_scores[lemma] > self.attention_score_threshold]
        selected_lemma_overall_scores = {}
        for lemma in lemma_greater_than_threshold_max_scores:
            selected_lemma_overall_scores[lemma] = lemma_overall_scores[lemma]
        selected_lemma_after_cutoff = sorted(selected_lemma_overall_scores, key=selected_lemma_overall_scores.get, reverse=True)[:self.maximum_dictionary_expansion]
        for lemma1 in selected_lemma_after_cutoff:
            for lemma2 in lemma_to_lemma_scores[lemma1]:
                if lemma_to_lemma_scores[lemma1][lemma2] > 0.0:
                    # node1_type = AmocNodeType.INFERRED 
                    # node2_type = AmocNodeType.INFERRED
                    # node1 = AmocNode(lemma1, node1_type)
                    # node2 = AmocNode(lemma2, node2_type)
                    node1 = self.amoc_graph.get_graph_node_from_text(lemma1)
                    node2 = self.amoc_graph.get_graph_node_from_text(lemma2)
                    self.amoc_graph.add_edge_both_ways(node1, (node2, lemma_to_lemma_scores[lemma1][lemma2]))
            
            # if spacy_doc_generated[-1].pos_ == "PUNCT" or len(list(spacy_doc_generated.sents)) <= 1:
            #     number_of_senteces = len(list(spacy_doc_generated.sents))
            #     self.analyze_last_sentences(text + generated_text, number_of_senteces, generation=True)
            # else:
            #     generated_text_until_last_sentence = " ".join([str(sent) for sent in list(spacy_doc_generated.sents)[:-1]])
            #     number_of_senteces = len(list(spacy_doc_generated.sents)) - 1
            #     self.analyze_last_sentences(text + generated_text_until_last_sentence, number_of_senteces, generation=True)

    def amoc_analyze_with_expansion(self, text: str):
        text = re.sub(' +', ' ', text.strip())
        full_doc = self.spacy_model(text)
        for i in range(len(list(full_doc.sents))):
            
            if i == 0:
                selected_text = str(list(full_doc.sents)[i])
                self.analyze_only_one_sentence(selected_text)
                text_components_char_start = self.get_graph_lookback_and_current_text_char_start_dict("", (list(full_doc.sents)[i]))
                self.generate_text_and_add_to_amoc_graph(selected_text, text_components_char_start)
                self.amoc_graph.activate_and_deactivate_nodes()
                self.amoc_graph.save_graph_to_pickle(i, self.folder_to_save_graphs)
                # self.amoc_graph.draw_graph_top_n(i)
                continue
            
            self.amoc_graph.decay_edges_with_percentage(self.weigth_decay_percatage)
            
            lookback_sents = list(full_doc.sents)[max(i-self.number_of_sentences_lookback,0):i]
            current_sent = list(full_doc.sents)[i]
            # remaining_active_concepts_from_graph = self.get_active_concepts_from_graph_that_are_not_in_text(selected_sents)
            remaining_active_concepts_from_graph = self.get_active_concepts_from_graph(only_text_based=False)
            graph_text = ""
            if remaining_active_concepts_from_graph:
                graph_text = " ".join(remaining_active_concepts_from_graph) + ". "
            lookback_sents_text = " ".join([str(sent) for sent in lookback_sents])
            current_sent_text = str(current_sent)
            to_be_processed_text = graph_text + lookback_sents_text + " " + current_sent_text
            text_components_char_start = self.get_graph_lookback_and_current_text_char_start_dict(graph_text, lookback_sents)
            self.analyze_last_sentences(to_be_processed_text, text_components_char_start, number_of_last_sentences=self.number_of_sentences_lookback, generation=False)
            
            # we get the new active concepts from the graph after reading the current sentence and THEN we generate text
            # remaining_active_concepts_from_graph_after_reading_current_sentence = self.get_active_concepts_from_graph_that_are_not_in_text(selected_sents)
            remaining_active_concepts_from_graph_after_reading_current_sentence = self.get_active_concepts_from_graph()
            graph_text = ""
            if remaining_active_concepts_from_graph_after_reading_current_sentence:
                graph_text = " ".join(remaining_active_concepts_from_graph_after_reading_current_sentence) + ". "
            lookback_sents = list(full_doc.sents)[max(i-self.number_of_sentences_lookback,0):i+1]
            to_be_generated_from_text = graph_text + " ".join([str(sent) for sent in lookback_sents])
            text_components_char_start = self.get_graph_lookback_and_current_text_char_start_dict(graph_text, lookback_sents)
            self.generate_text_and_add_to_amoc_graph(to_be_generated_from_text, text_components_char_start)
            #print(self.amoc_graph.get_max_active_nodes_by_score())
            print(self.amoc_graph.get_max_active_nodes_by_score(use_aoe=self.use_aoe))
            # print(self.amoc_graph.get_top_n_nodes_by_score(10))
            print(self.amoc_graph.get_top_n_nodes_by_score(10, use_aoe=self.use_aoe))
            self.amoc_graph.activate_and_deactivate_nodes()
            self.amoc_graph.save_graph_to_pickle(i, self.folder_to_save_graphs)
            # self.amoc_graph.draw_graph_top_n(i)
    
    
    def get_graph_lookback_and_current_text_char_start_dict(self, graph_text: str, lookback_sents: List[Span]):
        lookback_sents_start_characters = []
        for i in range(len(lookback_sents)):
            if i == 0:
                lookback_sents_start_characters.append(0)
            else:
                lookback_sents_start_characters.append(lookback_sents_start_characters[i-1] + len(str(lookback_sents[i-1])))
        return {
            "graph_start": 0,
            "lookback_sents_starts": [x + len(graph_text) for x in lookback_sents_start_characters],
            "current_start": len(graph_text) + lookback_sents_start_characters[-1] + len(str(lookback_sents[-1]))
        }
            
    
    def get_active_concepts_from_graph(self, only_text_based=True):
        top_nodes = list(self.amoc_graph.get_max_active_nodes_by_score(only_text_based=only_text_based, use_aoe=self.use_aoe).keys())
        return [x.text for x in top_nodes]
    

    def get_active_concepts_from_graph_that_are_not_in_text(self, selected_sents, only_text_based=True):
        top_nodes = list(self.amoc_graph.get_max_active_nodes_by_score(only_text_based=only_text_based, use_aoe=self.use_aoe).keys())
        remaining_words = []
        for node in top_nodes:
            is_word_found = False
            for sent in selected_sents[:-1]:
                for sent_word in sent:
                    if sent_word.lemma_ == node.text:
                        is_word_found = True
                        break
            if not is_word_found:
                remaining_words.append(node.text)
        return remaining_words
                    

    def amoc_analyze_only_text(self, text: str):
        full_doc = self.spacy_model(text)
        for i in range(len(list(full_doc.sents))):
            if i == 0:
                selected_text = str(list(full_doc.sents)[i])
                self.analyze_only_one_sentence(selected_text)
                continue
            selected_sents = list(full_doc.sents)[max(i-self.number_of_sentences_lookback,0):i+1]
            selected_text = " ".join([str(sent) for sent in selected_sents])
            top_nodes = list(self.amoc_graph.get_max_active_nodes_by_score().keys())
            remaining_words = []
            for node in top_nodes:
                is_word_found = False
                for sent in selected_sents[:-1]:
                    for sent_word in sent:
                        if sent_word.lemma_ == node.text:
                            is_word_found = True
                            break
                if not is_word_found:
                    remaining_words.append(node.text)
            if remaining_words:
                selected_text = " ".join(remaining_words) + ". " + selected_text
            print(selected_text)
            self.analyze_last_sentences(selected_text, number_of_last_sentences=1, generation=False)
            print(self.amoc_graph.get_max_active_nodes_by_score())
        

if __name__ == "__main__":
    # amoc_analyze_only_text(look_back=2, max_active_concepts=4)
    text = "A young knight rode through the forest. The knight was unfamiliar with the country. Suddenly, a dragon appeared. The dragon was kidnapping a beautiful princess. The knight wanted to free the princess. The knight wanted to marry the princess. The knight hurried after the dragon. They fought for life and death. Soon, the knight's armor was completely scorched. At last, the knight killed the dragon. The knight freed the princess. The princess was very thankful to the knight. She married the knight."
    amoc = AMOCBert(bert_model_name='gpt2', spacy_model_name='en_core_web_lg', generator_model_name='gpt2',
                        number_of_sentences_lookback=1, maximum_active_concepts=5,
                        maximum_dictionary_expansion=7, attention_score_threshold=0.3, generation_imagination_factor=2,
                        weigth_decay_percatage=0.1, use_aoe=True)
    amoc.amoc_analyze_with_expansion(text)