import argparse
import os
from chemnlp.utils.process_doc import ProcessDoc

parser = argparse.ArgumentParser(
    description="Natural language processing for chemical data."
)

parser.add_argument(
    "--file_path",
    default="./",
    help="Filepath for text file parsing.",
)

if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    with open(args.file_path, "r") as f:
        txt = f.read().splitlines()
    pdoc = ProcessDoc(text=txt)
    print(pdoc.get_chemical_formulae())
