from connect import AMOCBert
import os


if __name__ == "__main__":
    for i in range(1, 20):
        low_coh_text = ""
        with open(f"datasets/low_high_cohesion/{i}/low.txt", "r", encoding="utf-8") as f:
            low_coh_text = f.read()
        high_coh_text = ""
        with open(f"datasets/low_high_cohesion/{i}/high.txt", "r", encoding="utf-8") as f:
            high_coh_text = f.read()
        amoc = AMOCBert(bert_model_name='gpt2', spacy_model_name='en_core_web_lg', generator_model_name='gpt2',
                        number_of_sentences_lookback=1, maximum_active_concepts=5,
                        maximum_dictionary_expansion=7, attention_score_threshold=0.3, generation_imagination_factor=2,
                        weigth_decay_percatage=0.1, use_aoe=True, folder_to_save_graphs=f"saved_graphs/low_high_cohesion/{i}/low")
        amoc.amoc_analyze_with_expansion(low_coh_text)
        
        amoc = AMOCBert(bert_model_name='gpt2', spacy_model_name='en_core_web_lg', generator_model_name='gpt2',
                        number_of_sentences_lookback=1, maximum_active_concepts=5,
                        maximum_dictionary_expansion=7, attention_score_threshold=0.3, generation_imagination_factor=2,
                        weigth_decay_percatage=0.1, use_aoe=True, folder_to_save_graphs=f"saved_graphs/low_high_cohesion/{i}/high")
        amoc.amoc_analyze_with_expansion(high_coh_text)