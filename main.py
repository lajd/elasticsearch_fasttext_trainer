#!/usr/bin/env python3
import argparse
from trainer import FastTextTrainer

parser = argparse.ArgumentParser(
    description='Command line options for training/updating a FastText model from a field in elasticsearch'
)

parser.add_argument('index_name', type=str, help='Name of elasticsearch index')
parser.add_argument('es_text_field', type=str, help='Name of elasticsearch field to train over')
parser.add_argument('save_model_pth', type=str, help='Path to binary model file')
parser.add_argument('num_training_epochs', type=int, help='Path to binary model file')
parser.add_argument('--must_have_fields', type=str, required=False, nargs='*', default=[],
                    help='Fields which must be contained in the document to be elligible for training')
parser.add_argument('--must_not_have_fields', type=str, required=False, nargs='*', default=[],
                    help='Fields which must not be contained in the document to be elligible for training')
parser.add_argument('--max_vocab_size', type=int, default=None, required=False, help='Maximum size of vocab. None if no limit.')
parser.add_argument('--use_analyzed_field', type=bool, default=False, required=False, help='Whether to train model on an analyzed field. The default analyzer stems and lowers each token.')
parser.add_argument('--min_count', type=int, default=2, required=False, help='Minimum number of occurences of a given term to be added to the vocabulary')
parser.add_argument('--init_from_pretrained', type=bool, required=False, help='Whether to resume training a binary model')
parser.add_argument('--pretrained_path', type=str, required=False, help='Path to binary model file')
parser.add_argument('--bsize', type=int, required=False, default=128, help='Batch size when reading from Elasticsearch')


if __name__ == '__main__':
    args = parser.parse_args()
    kwargs = vars(args)
    fastText = FastTextTrainer(**kwargs)
    fastText.train()
