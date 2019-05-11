#!/usr/bin/env python3
from elasticsearch import Elasticsearch

from segtok import tokenizer


class TextFieldIterator:
    def __init__(self, index_name, es_field_name, must_have_fields=None, must_not_have_fields=None, bsize=100):
        self.es_utility = ESUtility(index_name, bsize)
        self.es_field_name = es_field_name
        self.must_have_fields = must_have_fields or []
        self.must_not_have_fields = must_not_have_fields or []

    @staticmethod
    def tokenize_text(text, do_lower=True):
        """ Alter this method for custom tokenizer. By default, use segtok tokenizer.
        """
        if isinstance(text, list) and isinstance(text[0], str):
            # Allow handling string-array fields in ES
            text = ' '.join(text)
        if do_lower:
            text = text.lower()

        return tokenizer.word_tokenizer(text)

    def sentences_iterator(self, do_lower=True):
        # Do a full pass over the data set
        for batch in self.es_utility.scroll_indexed_data(self.es_field_name, self.must_have_fields, self.must_not_have_fields):
            for d in batch:
                source = d['_source']
                text = source[self.es_field_name]
                tokens = self.tokenize_text(text, do_lower)
                yield tokens

    def __iter__(self):
        return self.sentences_iterator()


class ESUtility:
    def __init__(self, index_name, bsize=100):
        self.index_name = index_name
        self.es = Elasticsearch([{'host': 'localhost', 'port': 9200}], timeout=60, request_timeout=60)
        assert self.es.ping(), 'Cannot ping elasticsearch on localhost:9200'
        print('Successfully pinged elasticsearch')
        # Check index exists
        if not self.es.indices.exists(index=index_name):
            raise RuntimeError('Index {} not found'.format(index_name))
        self.bsize = bsize

    def scroll_indexed_data(self, field_name, fields_must_exist=None, fields_must_not_exist=None, log_every_n_batches=500):
        if fields_must_exist is None: fields_must_exist = []
        if fields_must_not_exist is None: fields_must_not_exist = []
        fields_must_exist = list(set(fields_must_exist + [field_name]))
        fields_must_not_exist = list(set(fields_must_not_exist))
        # Return data in order
        body = {
            "_source": [field_name],
            "query": {
                "bool": {
                    "must": [{"exists": {"field": f}} for f in fields_must_exist],
                    "must_not": [{"exists": {"field": f}} for f in fields_must_not_exist]
                }
            },
        }
        data_scroll = self.es.search(
            index=self.index_name,
            scroll='5m',
            size=self.bsize,
            body=body,
        )
        sid = data_scroll['_scroll_id']
        scroll_size = len(data_scroll['hits']['hits'])
        bcount = 0
        while scroll_size > 0:
            yield (data_scroll['hits']['hits'])
            data_scroll = self.es.scroll(scroll_id=sid, scroll='5m')
            # Update the scroll ID
            sid = data_scroll['_scroll_id']
            # Get the number of results that returned in the last scroll
            scroll_size = len(data_scroll['hits']['hits'])
            bcount += 1
            if bcount % log_every_n_batches == 0:
                print('Processed {} batches ({} documents)'.format(bcount, bcount*self.bsize))
