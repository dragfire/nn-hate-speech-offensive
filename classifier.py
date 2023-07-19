import torch
from datautil import line_to_indices
import argparse

model = torch.load('model.lstm.pt')
model_meta = torch.load('model_meta.pt')

def classify(line):
    x = line_to_indices(model_meta['word_to_ix'], line).reshape(1, 1, -1)
    x = x.to(model_meta['device'])
    return model_meta['class_labels'][model(x).topk(1)[1].item()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classifier")
    parser.add_argument("--text", type=str, default="hello", help="Text to classify")
    args = parser.parse_args()
    
    print(f'\nGiven: "{args.text}"')
    print(f'Result: {classify(args.text)}')

