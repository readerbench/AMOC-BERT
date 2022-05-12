# from connect import AmocGraph
from os import listdir
import os


def read_scott_category_files(path):
    folder_contents = {}
    for file in listdir(path):
        with open(path + file, "r", encoding="unicode_escape") as f:
            content = f.read()
            text_number = file.split(".")[0][-3:]
            folder_contents[text_number] = content
    return folder_contents


def read_scott_files():
    advanced = read_scott_category_files("datasets/simplified_texts/advanced/")
    elementary = read_scott_category_files("datasets/simplified_texts/elementary/")
    intermediate = read_scott_category_files("datasets/simplified_texts/intermediate/")

    grouped = {}
    for key in advanced.keys():
        grouped[key] = {
            "advanced": advanced[key],
            "elementary": elementary[key],
            "intermediate": intermediate[key]
        }
    
    return grouped


def create_saved_graphs_folder_structure():
    grouped = read_scott_files()
    for key in grouped:
        for category in grouped[key]:
            os.makedirs(f"saved_graphs/simplified_texts/{key}/{category}")
    

def create_measurements_folder_structure():    
    grouped = read_scott_files()
    for key in grouped:
        for category in grouped[key]:
            os.makedirs(f"measurements/simplified_texts/{key}/{category}")


if __name__ == "__main__":
    pass