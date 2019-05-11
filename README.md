# elasticsearch_fasttext_trainer
Train a new, or update an existing FastText embedding model from an elasticsearch field (CPU only) </br>
Uses Gensim's wrapper over Facebook's FastText. </br>
https://radimrehurek.com/gensim/models/fasttext.html </br>
https://github.com/facebookresearch/fastText </br>
</br>

## Requirements
1) Running ES service on localhost:9200</br>

</br>

## Usage
This script can be used to train or update a FastText embedding model (https://github.com/facebookresearch/fastText) from a field in elasticsearch.
Given an running elasticsearch service on localhost:9200 with existing index <index_name>, we can:</br>
</br>
Create and train a new FastText model from scratch, from vocabulary found in <es_field>. Model will be saved to <save_model_pth>, and trained for <num_trianing_epochs> epochs. </br>
</br>

`python main.py <es_index_name> <es_field_name> <save_model_pth> <num_training_epochs> --[options]` </br>

</br>

Update an existing model by adding new vocabulary found in <es_field>, and continue training over <es_field>. This is the typical use case, and can 
be used to fine-tune model weights to your dataset. For example, download and unzip the pretrained FastText model from https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip.
Assuming the above `.bin` model is found in <pretrained_path>, to update vocab and fine-tune weights to <es_field>, do: </br>
</br>

`python main.py <es_index_name> <es_field_name> <save_model_pth> <num_training_epochs> --init_from_pretrained True --pretrained_path <pretrained_path> --[options]` </br>

</br>

## Additional parameters

--must_have_fields      --> Fields which must be contained in the document to be elligible for training </br>
--must_not_have_fields  --> Fields which must not be contained in the document to be elligible for training</br>
--bsize                 --> Batch size when reading from Elasticsearch</br>

</br>

## Tests
pytest tests.py </br>
