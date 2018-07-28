from typing import Tuple, List
import numpy as np


class CapeMachineReaderModelInterface:
    """Machine reader models should implement this interface. If model contributors implement this
    interface, then Cape should be able to seamlessly use the model (assuming their tokenization is consistent...)
    """

    def tokenize(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        """Tokenize text into word tokens. Return both a list of tokens, and
        a list of start, end character indices of the tokens in the input text

        :param text: text to tokenize, a string
        :return: list of string tokens, list of start, end character index tuples
        """
        raise NotImplementedError()

    def get_document_embedding(self, text: str) -> np.ndarray:
        """Embed a document into the highest question-independent space.

        Computing document embeddings is usually the most intensive part of the computation graph.
        We require the graph to be split in two parts for production environemnts. Usually many
        questions will be asked to a single document. Computing the document representation once
        rather than for every question greatly improves performance. This can be thought of as "caching"
        a high-level embedding of the document. This method takes a document as a string,
        then returns a 2d array of (N_words, dimension) document representation

        :param text: document to embed
        :return: 2d array (N_words, dimensions)
        """
        raise NotImplementedError()

    def get_logits(self, question: str, document_embedding: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get logit (unnormalised) scores for each token in a document for an inputted question.

        return two np arrays, where element i is the score for an answer starting or ending at token i
        for the inputted document given the question

        :param question: question string
        :param document_embedding: embedded document representation (as produced by self.get_document_embedding)
        :return: tuple of (start answer span scores, end answer span scores)
        """
        raise NotImplementedError()
