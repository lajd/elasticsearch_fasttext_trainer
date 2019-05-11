#!/usr/bin/env python3
import os

from es_utils import ESUtility
from trainer import FastTextTrainer

import nltk
from nltk.corpus import brown
from elasticsearch import Elasticsearch, helpers
from gensim.models.fasttext import FastText

nltk.download('brown')

# Define indexing parameters
INDEX_NAME = 'test'
FIELD_NAME = 'data'
# Client
es = Elasticsearch([{'host': 'localhost', 'port': 9200}], timeout=5, request_timeout=5)


def test_es_connection():
    assert es.ping()


def test_train_new_model():
    """
    Test training the model from scratch
    """
    save_model_pth = 'temp.model'
    # Index some data into ES
    reindex(20)
    # Create the fasttext model
    fasttext_trainer = FastTextTrainer(
        INDEX_NAME, FIELD_NAME, save_model_pth=save_model_pth, num_training_epochs=1, init_from_pretrained=False
    )
    # Train the model and write to the output path
    fasttext_trainer.train()
    assert os.path.exists(save_model_pth)
    # Load the model to validate vectors were added
    created_model = FastText.load(save_model_pth)
    assert len(created_model.wv.vectors) > 0
    # Remove the created model files
    os.system('rm {}*'.format(save_model_pth))


def test_continued_training_from_existing_model():
    """
    Test training the model from scratch
    """
    save_model_pth = 'temp.model'
    # Create a very small model, resulting in a small vocabulary
    reindex(num_sentences=5)
    fasttext_trainer = FastTextTrainer(
        INDEX_NAME, FIELD_NAME, save_model_pth=save_model_pth, num_training_epochs=1, init_from_pretrained=False
    )
    fasttext_trainer.train()
    assert os.path.exists(save_model_pth)
    previous_vocab = set(fasttext_trainer.model.wv.vocab.keys())
    # Re-index more data (resulting in larger vocabulary)
    reindex(num_sentences=50, create_new_index=False)
    # Continue training the model
    fasttext_trainer = FastTextTrainer(
        INDEX_NAME, FIELD_NAME, save_model_pth=save_model_pth, num_training_epochs=1,
        init_from_pretrained=True, pretrained_path=save_model_pth
    )
    fasttext_trainer.train()
    # Validate the updated model
    updated_model = FastText.load(save_model_pth)
    new_vocab = set(updated_model.wv.vocab.keys())
    assert len(new_vocab) > len(previous_vocab)
    assert previous_vocab.issubset(new_vocab)
    # Remove the created model files
    os.system('rm {}*'.format(save_model_pth))


class ExampleTextIterator:
    """ Example data to index """
    def __init__(self, n_examples):
        self.n_examples = n_examples

    def __iter__(self):
        c = 0
        for sent_toks in brown.sents():
            sent = ' '.join(sent_toks)
            if c == self.n_examples:
                raise StopIteration
            yield sent
            c += 1


def reindex(num_sentences, create_new_index=True):
    print('Reindexing')
    # Re-index
    if create_new_index:
        if es.indices.exists(INDEX_NAME):
            es.indices.delete(INDEX_NAME)
        es.indices.create(INDEX_NAME, {})
    es_utility = ESUtility(index_name=INDEX_NAME)
    # Index NUM_TEST_EXAMPLES data under the field we are interested in
    index_text_data(es_utility.es, FIELD_NAME, INDEX_NAME, ExampleTextIterator(num_sentences))
    return es_utility


def index_text_data(es, field_name, index_name, text_iterator, bsize=100):
    """
    Args:
        es (Elasticsearch client object): ES client
        field_name (str): Name of field to index text under
        index_name (str): Name of ES index to put data in
        text_iterator (iterable): iterable of text to index
        bsize (int): write batch size
    """
    id_counter = 0
    chunk = []
    for text in text_iterator:
        chunk.append({"_op_type": "index", "_index": index_name, "_type": "_doc", field_name: text, '_id': id_counter})
        id_counter += 1
        if id_counter % bsize == 0:
            helpers.bulk(es, chunk, index=index_name, doc_type="_doc", refresh=True)
            chunk = []
    if len(chunk) > 0:
        helpers.bulk(es, chunk, index=index_name, doc_type="_doc", refresh=True)
    print('Finished indexing {} documents'.format(id_counter))

