

from transformers import pipeline
import argparse





if __name__ == "__main__":

    data = [
    {"prompt": "we create [MASK]", "ground_truth": "chemistry"},
    {"prompt":"See corresponding entries for ‘Transport [MASK] class(es” for the respective regulations in the tables above.", "ground_truth": "hazard"},
    {"prompt":"See corresponding entries for ‘Packing [MASK]” for the respective regulations in the tables above.", "ground_truth": "group"},
    {"prompt":"Safety  [MASK] Sheet", "ground_truth": "data"},
    {"prompt":"No mortality was observed. Inhalation-risk test (IRT: No mortality within 7 hours as shown in [MASK] studies.", "ground_truth":"animal"},
    {"prompt":"Not classified as a [MASK] good under transport regulations", "ground_truth":"dangerous"},
    {"prompt":"No [MASK] was observed.", "ground_truth": "mortality"},
    {"prompt":"The substances/groups of substances mentioned can be released in case of fire. Evolution of fumes/fog. Burning produces [MASK] and toxic fumes.","ground_truth":"harmful"},
    {"prompt":"Further [MASK] release measures:","ground_truth":"accidental"}
    ]
    fill_mask = pipeline(
    "fill-mask",
    model="recobo/chemical-bert-uncased",
    tokenizer="recobo/chemical-bert-uncased"
    )
    for data_point in data:
        prompt = data_point["prompt"]
        ground_truth = data_point["ground_truth"]
        prediction = (fill_mask(prompt)[0]["token_str"])
        print(f"Prompt: {prompt}\nPrediction: {prediction}\nGroundTruth: {ground_truth}\n\n")



