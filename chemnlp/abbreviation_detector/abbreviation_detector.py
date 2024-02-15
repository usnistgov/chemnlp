import spacy

from scispacy.abbreviation import AbbreviationDetector
import argparse
import sys


parser = argparse.ArgumentParser(description="ChemNLP package.")
parser.add_argument(
    "--text",
    default="Spinal and bulbar muscular atrophy (SBMA) is an \
            inherited motor neuron disease caused by the expansion \
            of a polyglutamine tract within the androgen receptor (AR). \
            SBMA can be caused by this easily.",
 
)



if __name__ == "__main__":
    # python generate_data.py
    # generate_data()
    # classify(csv_path='cond_mat.csv')
    args = parser.parse_args(sys.argv[1:])

    text = args.text

    nlp = spacy.load("en_core_sci_sm")

    # Add the abbreviation pipe to the spacy pipeline.
    nlp.add_pipe("abbreviation_detector")

    doc = nlp(text)

    print("Abbreviation", "\t", "Definition")
    for abrv in doc._.abbreviations:
        print(f"{abrv} \t ({abrv.start}, {abrv.end}) {abrv._.long_form}")



