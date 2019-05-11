#!/usr/bin/env python3
import time

from es_utils import TextFieldIterator

from gensim.models.fasttext import FastText
from gensim.models.fasttext import load_facebook_model


class FastTextTrainer:
    def __init__(self, index_name, es_text_field, save_model_pth, num_training_epochs, save_vocab_model_pth=None,
                 init_from_pretrained=False, pretrained_path=None, bsize=128, emb_dim=300, window=5,
                 min_count=2, workers=12, max_vocab_size=None, must_have_fields=None, must_not_have_fields=None):

        assert (not init_from_pretrained and not pretrained_path) or (init_from_pretrained and pretrained_path), \
            "If init_from_pretrained=True, pretrained_path must be provided"
        if must_have_fields is None: must_have_fields = []
        if must_not_have_fields is None: must_not_have_fields = []
        # General parameters
        self.index_name = index_name
        self.es_text_field = es_text_field
        self.bsize = bsize
        self.save_model_pth = save_model_pth
        self.num_training_epochs = num_training_epochs
        self.save_vocab_model_pth = save_vocab_model_pth
        self.must_have_fields = must_have_fields + [es_text_field]
        self.must_not_have_fields = must_not_have_fields
        # Training parameters
        self.emb_dim = emb_dim  # May be updated below if using a pretrained model
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.max_vocab_size = max_vocab_size

        # Initialize model
        if init_from_pretrained and pretrained_path:
            try:
                self.model = load_facebook_model(pretrained_path)
            except NotImplementedError:
                self.model = FastText.load(pretrained_path)
            except Exception as e:
                raise RuntimeError("Fasttext model is neither facebook nor gensim model: {}".format(e))

            self.emb_dim = self.model.vector_size
            print('Ignoring arg `emb_dim` since loading a pretrained model. Emb_dim is set to {}'.format(self.emb_dim))
            # Update parameters
            self.model.workers = self.workers
            # Start iterating corpus and building vocab
            print('Updating vocabulary with first pass over corpus')
            self.model.build_vocab(sentences=TextFieldIterator(index_name, es_text_field, self.must_have_fields, self.must_not_have_fields, bsize=self.bsize), update=True)
        else:
            # More parameters can be exposed when creating a new model;
            # https://radimrehurek.com/gensim/models/fasttext.html#gensim.models.fasttext.FastText
            # Instantiate model
            self.model = FastText(size=emb_dim, window=window, min_count=min_count, workers=workers)
            # Start iterating corpus and building vocab
            self.model.build_vocab(sentences=TextFieldIterator(index_name, es_text_field, self.must_have_fields, self.must_not_have_fields, bsize=self.bsize))
        if self.save_vocab_model_pth:
            print('Finished creating vocab; saving vocab model')
            self.model.save(self.save_vocab_model_pth)

    def train(self):
        # Train model
        t1 = time.time()
        print('Beginning training')
        self.model.train(
            sentences=TextFieldIterator(self.index_name, self.es_text_field, self.must_have_fields, self.must_not_have_fields, bsize=self.bsize),
            total_examples=self.model.corpus_count,
            epochs=self.num_training_epochs,
        )
        t2 = time.time()
        print('Ended training in {}s'.format(round(t2-t1)))
        # Save to path
        print('Saving model')
        self.model.save(self.save_model_pth)
