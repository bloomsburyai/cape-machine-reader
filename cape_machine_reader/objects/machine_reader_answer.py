from typing import Tuple


class MachineReaderAnswer:
    """
    MachineReaderAnswer is expected to have a:
    - text:                     string that is not None
    - score_reader:             float in [0.0, 1.0]
    - score_answer_in_document:  float in [0.0, 1.0]
    - span:                     Tuple[int, int] = (None, None)
    """
    text: str
    score_reader: float
    score_answer_in_document: float
    span: Tuple[int, int]

    def __init__(self, text: str, span: Tuple[int, int], long_text: str, long_text_span: Tuple[int, int],
                 score_reader: float, score_answer_in_document: float):
        assert text is not None
        assert long_text is not None
        assert 0.0 <= score_reader <= 1.0
        assert 0.0 <= score_answer_in_document <= 1.0
        assert len(span) == 2
        assert span[0] is None or isinstance(span[0], int)
        assert span[1] is None or isinstance(span[1], int)
        if isinstance(span[0], int):
            assert span[1] is not None
        if isinstance(span[1], int):
            assert span[0] is not None
        assert len(long_text_span) == 2
        assert long_text_span[0] is None or isinstance(long_text_span[0], int)
        assert long_text_span[1] is None or isinstance(long_text_span[1], int)
        if isinstance(long_text_span[0], int):
            assert long_text_span[1] is not None
        if isinstance(long_text_span[1], int):
            assert long_text_span[0] is not None

        self.text = text
        self.span = span
        self.long_text = long_text
        self.long_text_span = long_text_span
        self.score_reader = score_reader
        self.score_answer_in_document = score_answer_in_document
