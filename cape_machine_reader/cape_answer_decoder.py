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

from interval import interval
import numpy as np
from typing import List, Tuple, Iterable
from dataclasses import dataclass


def find_answer_spans(y1_list: List[float], y2_list: List[float]) -> Iterable[Tuple[Tuple[int, int], float]]:
    """Efficiently produce answer spans from start answer probabilities and end answer probabilities for words.

    :param y1_list: list of floats - a probability distribution (i.e. no negative values, and must sum to 1) over
       the tokens of a document. y1_list[i] is the probability an answer to the question starts at token i.
    :param y2_list: list of floats - a probability distribution (i.e. no negative values, and must sum to 1) over
       the tokens of a document. y2_list[i] is the probability an answer to the question ends at token i.
    :return an iterable of answer spans and scores, in descending order. The answer span is the index
        of the start word and the index of the end word
    """
    EPSILON = 1e-6  # Minimum sum of all probabilities to continue searching
    RESET_RANGE = 1  # Range around answer span to reset probabilities for. Set to 0 for exact span
    already_used = interval()
    remaining_continuations = int(1e2)
    while sum(y1_list) > EPSILON:
        cummax_y_start = np.maximum.accumulate(y1_list)

        # Precompute the indices of the locations where cummax is updated
        # Get the indices of the location where the cumulative max is equal to the maximum
        cummax_ind = np.nonzero(y1_list == cummax_y_start)[0]
        # Accumulate (similar to above)
        cumargmax = np.zeros_like(y1_list, dtype=np.int)
        cumargmax[cummax_ind] = cummax_ind
        cumargmax_y_start = np.maximum.accumulate(cumargmax)

        opt_pos_start = 0
        opt_pos_end = 0
        highest_max = y1_list[opt_pos_start] * y2_list[opt_pos_end]
        for i in range(1, len(y2_list)): # Need to start from the second element (i = 1) as we have predefined i = 0

            cur_highest_start_index = cumargmax_y_start[i]
            end_prob = y2_list[i]
            # Decay end prob by how far away it is from the current highest start position
            # Decay by proportion_to_decay of value for each word away
            proportion_to_decay = 0.01
            end_prob = max(0, end_prob * (1 - proportion_to_decay * (i - cur_highest_start_index)))

            cur_max = cummax_y_start[i] * end_prob # (Highest start prob seen so far) x (current end prob)
            if cur_max > highest_max:
                highest_max = cur_max
                # opt_pos_start = np.argmax(y1_list[:i+1])
                opt_pos_start = cumargmax_y_start[i]
                opt_pos_end = i
        span_word_indices = (opt_pos_start, opt_pos_end)
        score = y1_list[opt_pos_start] * y2_list[opt_pos_end]

        # Reset the selected range of index values
        range_start = max(0, opt_pos_start - RESET_RANGE)
        range_end = min(len(y1_list), opt_pos_end + RESET_RANGE)
        y1_list[range_start:range_end] = 0.0
        y2_list[range_start:range_end] = 0.0
        if remaining_continuations > 0:
            range_interval = interval[range_start,range_end]#create new interval object
            if already_used & range_interval:#check if the current answer contains a previous answer
                remaining_continuations -= 1
                continue
            else:
                already_used |=range_interval#update already used intervals

        # Yield the answer span indices and score
        yield span_word_indices, score


def softmax(logits) -> np.array:
   """Compute softmax values for each sets of scores in logits."""
   m = np.max(logits)
   return np.exp(logits - m) / np.sum(np.exp(logits - m), axis=0)


@dataclass
class AnswerSpan:
    answer_text: str
    character_indices: Tuple[int, int]
    word_indices: str
    long_answer_text: str
    long_character_indices: Tuple[int, int]
    long_word_indices: Tuple[int, int]
    score: float


def find_best_spans(context: str,
                    context_offsets: List[List[int]],
                    y1_list: List[float],
                    y2_list: List[float],
                    top_k: int,
                    long_text_expansion_in_words: int=20) -> Iterable[AnswerSpan]:
    """Find the top K answers in a document from probability distributions over the tokens in a document
    for the start and ending positions of answers


    :param context: The raw string of the document
    :param context_offsets: The character indices of the tokens in the document
    :param y1_list: a probability distribution (i.e. no negative values, and must sum to 1) over
       the tokens of a document. y1_list[i] is the probability an answer to the question starts at token i.
    :param y2_list: a probability distribution (i.e. no negative values, and must sum to 1) over
       the tokens of a document. y1_list[i] is the probability an answer to the question end at token i.
    :param top_k: The number of answers to decode
    :param long_text_expansion_in_words: How many words either side of the answer to include in the "answer with context"
    :return: AnswerSpan objects containing the data required to produce answers for downstream tasks
    """
    counter = 0
    y1_list = y1_list.copy()
    y2_list = y2_list.copy()
    iterator = find_answer_spans(y1_list, y2_list)
    while counter < top_k:
        counter += 1
        answer_word_indices, score = next(iterator)
        answer_char_indices = (context_offsets[answer_word_indices[0]][0], context_offsets[answer_word_indices[1]][1])
        answer_text = context[answer_char_indices[0]:answer_char_indices[1]]

        # Generate long text
        long_text_word_indices = (max(0, answer_word_indices[0] - long_text_expansion_in_words),
                                  min(len(context_offsets) - 1, answer_word_indices[1] + long_text_expansion_in_words))
        long_text_char_indices = (context_offsets[long_text_word_indices[0]][0], context_offsets[long_text_word_indices[1]][1])
        long_text = context[long_text_char_indices[0]:long_text_char_indices[1]]
        yield AnswerSpan(answer_text, answer_char_indices, answer_word_indices,
                         long_text, long_text_char_indices, long_text_word_indices, score)
