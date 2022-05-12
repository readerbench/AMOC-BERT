from operator import index
from transformers import pipeline, AutoTokenizer, AutoModel, AutoConfig, set_seed
from transformers import DistilBertTokenizer, TFDistilBertModel, DistilBertConfig
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import numpy as np


def mask_replacement():
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    MASK_TOKEN = tokenizer.mask_token
    unmasker = pipeline("fill-mask", model='distilbert-base-uncased')
    result = unmasker(f"Hello I'm a {MASK_TOKEN} model.")

    print(result)
    
def knight_knight():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    MASK_TOKEN = tokenizer.mask_token
    unmasker = pipeline("fill-mask", model='distilbert-base-uncased')
    text = "knight "
    for _ in range(10):
        input = text + MASK_TOKEN
        result = unmasker(input)
        print(f"Iteration {_ + 1}, text: {input}\n, result:\n{result}")
        text += "knight " 
        print()


def get_attentions():
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    # config = AutoConfig.from_pretrained('distilbert-base-uncased', output_hidden_states=True, output_attentions=True)
    # model = AutoModel.from_pretrained('distilbert-base-uncased', config=config)
    model = AutoModel.from_pretrained('distilbert-base-uncased')
    text = "I am in a clothing store. A girl comes to me and says: Hello, I am a model."
    tokenized_sequence = tokenizer(text, return_tensors="tf")
    print(tokenized_sequence)
    outputs = model(tokenized_sequence)
    print(outputs)
    

def online_exemple():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    config = DistilBertConfig.from_pretrained('distilbert-base-uncased', output_hidden_states=True, output_attentions=True)
    model = TFDistilBertModel.from_pretrained("distilbert-base-uncased", config=config)
    text = "I am in a clothing store. A girl comes to me and says: Hello, I am a model."
    encoded_input = tokenizer(text, return_tensors='tf')
    output = model(encoded_input)
    print(len(output))
    print(output[2])
    

def generate_example():
    sequence = "A young knight rode through the forest. The knight was unfamiliar with the country."
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = TFGPT2LMHeadModel.from_pretrained('gpt2')
    inputs = tokenizer(sequence, return_tensors='tf')
    outputs = model(inputs, output_attentions=True)
    print(tokenizer.decode([np.argmax(x) for x in outputs.logits[0]], skip_special_tokens=True))
    

def generate_example_with_pipeline():
    sequence = "A young knight rode through the forest. The knight was unfamiliar with the country."
    generator = pipeline('text-generation', model='gpt2')
    set_seed(42)
    results = generator(sequence, max_length=40, num_return_sequences=10, no_repeat_ngram_size=2,
                        return_full_text=False, early_stopping=True, do_sample=True, top_k=50, top_p=0.93) 
    from pprint import pprint
    pprint(results)
    
    
# online_exemple()
# knight_knight()
# generate_example()
generate_example_with_pipeline()
