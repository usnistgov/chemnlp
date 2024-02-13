

from transformers import pipeline
import argparse


import argparse
import sys


parser = argparse.ArgumentParser(description="ChemNLP package.")
parser.add_argument(
    "--text",
    default="See corresponding entries for ‘Transport [MASK] class(es” for the respective regulations in the tables above.",
 
)



if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])

    prompt = args.text


    fill_mask = pipeline(
    "fill-mask",
    model="recobo/chemical-bert-uncased",
    tokenizer="recobo/chemical-bert-uncased"
    )
    
    
    prediction = (fill_mask(prompt)[0]["token_str"])
    print(f"Prompt: {prompt}\nPrediction: {prediction}\n\n")



