# cape-machine-reader

Cape machine reader provides an interface between machine reading models and the rest of Cape,
and provides powerfully scaling answer decoding functionality for machine readers

## Who is this repo for?

This repo is helpful for those wishing to build and integrate their own machine reading models into Cape.
For low-level access to a Question answering functionality the [MachineReader objects](https://github.com/bloomsburyai/cape-machine-reader/blob/master/cape_machine_reader/cape_machine_reader_core.py),
object can be used, but most users may find the functionality exposed by [cape-responder](https://github.com/bloomsburyai/cape-responder) to be more complete and easy to
work with at a higher level.

## Who is this repo not for?

most users may find the functionality exposed by [cape-responder](https://github.com/bloomsburyai/cape-responder) to be more complete and easy to
work with at a higher level.

## Install:

To install as a site-package:

```
pip install --upgrade --process-dependency-links git+https://github.com/bloomsburyai/cape-document-qa
```

## Usage

[MachineReader objects](https://github.com/bloomsburyai/cape-machine-reader/blob/master/cape_machine_reader/cape_machine_reader_core.py) 
can be instantiated as follows (using the `cape-document-qa` model implementation):

```
>>> from cape_document_qa.cape_docqa_machine_reader import get_production_model_config, CapeDocQAMachineReaderModel
>>> from cape_machine_reader.cape_machine_reader_core import MachineReader, MachineReaderConfiguration

>>> mr_model_conf = get_production_model_config()
>>> mr_model = CapeDocQAMachineReaderModel(mr_model_conf)
>>> mr = MachineReader(mr_model)
>>> mr_answer_conf = MachineReaderConfiguration()
```


You can then use the machine reader in stand-alone mode:

```
>>> doc = "The Harry Potter series was written by J K Rowling"
>>> question = "Who wrote Harry Potter?"
>>> answers = mr.get_answers(mr_answer_conf, doc, question)
>>> print(next(answers).text)
'J K Rowling'
```

Or use it in a multistep workflow:

```
>>> doc = "The Harry Potter series was written by J K Rowling.\nHarry Potter is a Wizard"
>>> doc_1, doc_2 = doc.splitlines(True)
>>> question = " Who wrote Harry Potter?"
>>> logits_1, offsets_1 = mr.get_logits(doc_1, question)
>>> logits_2, offsets_2 = mr.get_logits(doc_2, question)
>>> answers = mr.get_answers_from_logits(
...    mr_answer_conf, [logits_1, logits_2], [offsets_1, offsets_2], doc)
>>> print(next(answers).text)
'J K Rowling'
```

And you can prepare document embeddings before hand. You could store these document embeddings which allows
for much faster question answering:

```
>>> doc = "The Harry Potter series was written by J K Rowling.\nHarry Potter is a Wizard"
>>> doc_1, doc_2 = doc.splitlines(True)
>>> doc_1_embedded = mr.get_document_embedding(doc_1)
>>> doc_2_embedded = mr.get_document_embedding(doc_2)
>>> # this does faster answering:
>>> logits_1, offsets_1 = mr.get_logits(doc_1, question, document_embedding=doc_1_embedded)
>>> logits_2, offsets_2 = mr.get_logits(doc_2, question, document_embedding=doc_2_embedded)
answers = mr.get_answers_from_logits(
...    mr_answer_conf, [logits_1, logits_2], [offsets_1, offsets_2], doc)
>>> print(next(answers).text)
'J K Rowling'
```

Note that documents shouldn't be longer than about 500 words for performant machine reading


## Integrating your own model:

If you have your own machine reading model and you want to make it compatible with cape, you
need to create a class that implements [cape_machine_reader_model.py](https://github.com/bloomsburyai/cape-machine-reader/blob/master/cape_machine_reader/cape_machine_reader_model.py).

The interface is designed to make as few assumptions about your model as possible, and to be as flexible as possible
whilst allowing performant machine reading. Models that fulfil this interface are wrapped by [MachineReader objects](https://github.com/bloomsburyai/cape-machine-reader/blob/master/cape_machine_reader/cape_machine_reader_core.py),
making them both powerful and useful for production environments.

Models are required to implement 3 simple methods:

### `tokenize`

The tokenize method should implement your model's NON-DESTRUCTIVE tokenization scheme.
Strings should be tokenized into lists of token strings, and a corresponding list of character start and end indices of the tokens.

### `get_document_embedding`

This method should accept a paragraph as a string (about 400-500 words).
Machine reading models embed the context paragraph and the question seperately, and then typically perform attention and project start-logit
scores and end-logit scores. 
This method should return the DEEPEST Question-independent document representation for your model.
This is so that Cape can cache this representation, which enables impressive speedups when multiple questions get
applied to the same document at test time.
The expected return type is a 2d numpy array, of shape (num_tokens, embedding dimension)

### `get_logits`

This method should accept a question as a string, and an embedded document context (generated by `get_document_embedding`), and should return 
two numpy arrays containing LOGIT scores for start and end spans in the document.

### Examples

ML developers could be inspired by [cape-document-qa](https://github.com/bloomsburyai/cape-document-qa) which contains a powerful
open-domain machine reader and implements `CapeMachineReaderModelInterface`.

An even simpler one that we use for testing (simply generates random logit scores) 
can be found here: [cape_machine_reader_model.py](cape_machine_reader/tests/test_machine_reader_model.py).


We are working on other models that have different characteristics (speed/scale/accuracy).

### Telling responder how to load your model

The dependency diagram of Cape is shown below, including a second Machine reading model for clarity.

![Dependencies Diagram](Dependencies_for_those_contributing_new_models.png)

[cape-responder](https://github.com/bloomsburyai/cape-responder) needs to be told how to load your new model.
Loading your model should be as simple as possible.

The code that imports cape-document-qa into responder is only 2 lines, and can be found [here](https://github.com/bloomsburyai/cape-responder/blob/7fa606ecdae623a2579475d737929d3f2059c1cc/cape_responder/responder_core.py#L44). You
can add your model loading logic adjacent to here to get Cape to use your model architecture.
