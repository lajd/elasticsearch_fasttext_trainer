#!/usr/bin/env python3
from elasticsearch import Elasticsearch

from segtok import tokenizer


class TextFieldIterator:
    def __init__(self, index_name, es_field_name, must_have_fields=None,
                 must_not_have_fields=None, bsize=100, use_analyzed_field=False):
        self.es_utility = ESUtility(index_name, bsize)
        self.use_analyzed_field = use_analyzed_field
        self.es_field_name = es_field_name
        self.must_have_fields = must_have_fields or []
        self.must_not_have_fields = must_not_have_fields or []

    @staticmethod
    def extract_tokens_from_termvectors(d, field_name):
        # Termvectors are already tokenized; Need to sort position
        tok_loc_tuples = []
        for tok, tok_attrs in d['term_vectors'][field_name]['terms'].items():
            tok_locs_elements = tok_attrs['tokens']
            for loc_element in tok_locs_elements:
                tok_loc_tuples.append((tok, loc_element['position']))
        tokens = [i[0] for i in sorted(tok_loc_tuples, key=lambda x: x[1])]
        return tokens

    def sentences_iterator(self, log_every=10000):
        # Do a full pass over the data set
        c = 0
        for batch in self.es_utility.scroll_indexed_data(self.es_field_name, self.must_have_fields,
                                                         self.must_not_have_fields, self.use_analyzed_field):
            for d in batch:
                if self.use_analyzed_field:
                    tokens = self.extract_tokens_from_termvectors(d, self.es_field_name)
                else:
                    source = d['_source']
                    text = source[self.es_field_name]
                    tokens = tokenizer.word_tokenizer(text)
                yield tokens
                c += 1
                if c % log_every == 0:
                    print("Processed {} documents".format(c))

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

    def scroll_indexed_data(self, field_name, fields_must_exist=None,
                            fields_must_not_exist=None, use_analyzed_field=False, log_every_n_batches=500):
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
            if not use_analyzed_field:
                hits = data_scroll['hits']['hits']
                yield hits
            else:
                # Make another request to yield termvectors
                documents = {'ids': [x['_id'] for x in data_scroll['hits']['hits']]}
                hits = self.es.mtermvectors(
                    body=documents, doc_type='_doc', index=self.index_name,
                    term_statistics=False, field_statistics=False
                )
                yield hits['docs']
            data_scroll = self.es.scroll(scroll_id=sid, scroll='5m')
            # Update the scroll ID
            sid = data_scroll['_scroll_id']
            # Get the number of results that returned in the last scroll
            scroll_size = len(data_scroll['hits']['hits'])
            bcount += 1
            if bcount % log_every_n_batches == 0:
                print('Processed {} batches ({} documents)'.format(bcount, bcount*self.bsize))
