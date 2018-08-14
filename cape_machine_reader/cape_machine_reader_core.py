# Copyright 2018 BLEMUNDSBURY AI LIMITED
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Iterable, Tuple, List, Optional
import numpy as np
from cape_machine_reader.objects.machine_reader_answer import MachineReaderAnswer
from cape_machine_reader.cape_answer_decoder import find_best_spans, softmax


class MachineReaderError(Exception):
    """Errors thrown when Machine Reading Goes wrong"""


class MachineReaderConfiguration:
    def __init__(self, threshold_reader: float = 0,
                 threshold_answer_in_document: float = 0,
                 top_k: int = 1
                 ):
        self.threshold_reader = threshold_reader
        self.threshold_answer_in_document = threshold_answer_in_document
        self.top_k = top_k


class MachineReader:

    def __init__(self, model):
        self.model = model

    @staticmethod
    def _combine_overlaps(text: str, before_overlap: str, after_overlap: str) -> str:
        return ''.join([before_overlap, text, after_overlap])

    def get_logits(self, text: str, question: str, before_overlap: str = '', after_overlap: str = '',
                   document_embedding: Optional[np.ndarray] = None) \
            -> Tuple[Tuple[np.array, np.array], Tuple[int, int]]:
        """Get unnormalised logit scores for a document and question

        :param text: document to ask question to
        :param question: question
        :param before_overlap: some text before the document
        :param after_overlap: some text after the document
        :param document_embedding: an optional document embedding for the document. If not
            supplied, the document embedding will be calculated here
        :return: two logit score distributions over the tokens of the document, for start and end span
            positions, and the number of tokens in the before_overlap and after_overlap strings
        """
        if self._count_tokens(text) == 0:
            raise MachineReaderError('Document cannot be empty : "{}"'.format(text))
        if self._count_tokens(question) == 0:
            raise MachineReaderError('Question cannot be empty : "{}"'.format(question))
        doc = self._combine_overlaps(text, before_overlap, after_overlap)
        if document_embedding is None:
            document_embedding = self.get_document_embedding(doc)

        n_total, n_before, n_text, n_after = map(self._count_tokens, [doc, before_overlap, text, after_overlap])
        if n_total != (n_before + n_text + n_after):
            raise MachineReaderError('Mismatch of N tokens: {} Expected, got {}'.format(n_total, n_before + n_text + n_after))
        return self.model.get_logits(question, document_embedding), (n_before, n_after)

    def _count_tokens(self, text):
        return len(self.model.tokenize(text)[0])

    def get_answers_from_logits(self,
                                configuration: MachineReaderConfiguration,
                                all_the_logits: List[Tuple[np.array, np.array]],
                                all_the_overlaps: List[Tuple[int, int]],
                                all_combined_texts: str,
                                ) -> Iterable[MachineReaderAnswer]:
        """Combine logit distributions from several documents and generate the highest scoring answers

        :param configuration: configuration object to control how answers are produced
        :param all_the_logits: list of (start_logit_scores, end_logit_scores) for the documents
        :param all_the_overlaps: list of (start token index, end token index) for where the begin_overlap
            and end_overap strings start in each document
        :param all_combined_texts: all the document strings as a single big string
        :return: iterable of machine reader answer objects
        """
        if len(all_the_logits) == 0:
            raise MachineReaderError('Need at least one block of logits')
        if len(all_the_overlaps) == 0:
            raise MachineReaderError('Need at least one block of overlaps')
        if len(all_the_overlaps) != len(all_the_logits):
            raise MachineReaderError('Overlaps and logits need to be the same length')

        logits_array_start = np.concatenate([
            logits[overlap_start:len(logits) - overlap_end] for (logits, _),  (overlap_start, overlap_end)
            in zip(all_the_logits, all_the_overlaps)
        ])
        logits_array_end = np.concatenate([
            logits[overlap_start:len(logits) - overlap_end] for (_, logits),  (overlap_start, overlap_end)
            in zip(all_the_logits, all_the_overlaps)
        ])
        if len(logits_array_start) != self._count_tokens(all_combined_texts):
            raise MachineReaderError(
                'logits length mismatch {} {}'.format(
                    len(logits_array_start), self._count_tokens(all_combined_texts)
            ))

        # Perform global softmax
        yp_start, yp_end = softmax(logits_array_start), softmax(logits_array_end)

        context_tokens, context_offsets = self.model.tokenize(all_combined_texts)
        answer_spans = find_best_spans(all_combined_texts, context_offsets, yp_start, yp_end, configuration.top_k)

        for answer_span in answer_spans:
            score_answer_in_document = 0.
            l1 = logits_array_start[answer_span.word_indices[0]]
            l2 = logits_array_end[answer_span.word_indices[1]]
            unnorm_score = l1 + l2

            if (answer_span.score >= configuration.threshold_reader and score_answer_in_document >=
                    configuration.threshold_answer_in_document):
                yield MachineReaderAnswer(text=answer_span.answer_text,
                                          span=answer_span.character_indices,
                                          long_text=answer_span.long_answer_text,
                                          long_text_span=answer_span.long_character_indices,
                                          score_reader=answer_span.score,
                                          score_answer_in_document=score_answer_in_document
                                          )
            else:
                break

    def get_document_embedding(self, text: str, before_overlap: str = '', after_overlap: str = '') -> np.array:
        """Generate a document embedding for a document. This document embedding can be stored/cached
        so that if more than one question gets asked to a document, work is not repeated

        :param text: text to embed
        :param before_overlap: small amount text before the text to embed (optional)
        :param after_overlap:small amount text after the text to embed (optional)
        :return: numpy 2d array of floats of shape (n tokens, embedding dimension)
        """
        if self._count_tokens(text) == 0:
            raise MachineReaderError('Document cannot be empty : "{}"'.format(text))
        return self.model.get_document_embedding(self._combine_overlaps(text, before_overlap, after_overlap))

    def get_answers(self, configuration: MachineReaderConfiguration, document_text: str, question: str) \
            -> Iterable[MachineReaderAnswer]:
        """Get answers from a document

        :param configuration: configuration object to control how answers are produced
        :param document_text: document to search for question answer
        :param question: question to ask to document
        :return: Iterable of machine reader answers, highest scoring first
        """
        all_logits, all_overlaps = self.get_logits(document_text, question)
        return self.get_answers_from_logits(
                configuration, [all_logits], [all_overlaps], document_text)

