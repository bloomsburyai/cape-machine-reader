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

import pytest
from cape_machine_reader.objects.machine_reader_answer import MachineReaderAnswer


def test_answer_creation():
    a = MachineReaderAnswer(text="test", span=(1, 5), long_text="long_test", long_text_span=(3, 9),
                            score_reader=0.5, score_answer_in_document=0.1)
    assert a.text == "test"
    assert a.span == (1, 5)
    assert a.long_text == "long_test"
    assert a.long_text_span == (3, 9)
    assert a.score_reader == 0.5
    assert a.score_answer_in_document == 0.1


def test_answer_creation_defaults():
    with pytest.raises(TypeError):
        MachineReaderAnswer(text='test', score_reader=0.5, score_answer_in_document=0.1)


def test_answer_invalid_text():
    with pytest.raises(AssertionError):
        MachineReaderAnswer(text=None, span=(1, 5), long_text="long_test", long_text_span=(3, 9),
                            score_reader=0.5, score_answer_in_document=0.1)

def test_answer_invalid_long_text():
    with pytest.raises(AssertionError):
        MachineReaderAnswer(text="test", span=(1, 5), long_text=None, long_text_span=(3, 9),
                            score_reader=0.5, score_answer_in_document=0.1)

def test_answer_invalid_score_reader():
    with pytest.raises(AssertionError):
        MachineReaderAnswer(text="test", span=(1, 5), long_text="long_test", long_text_span=(3, 9),
                            score_reader=20, score_answer_in_document=0.1)


def test_answer_invalid_score_answer_in_document():
    with pytest.raises(AssertionError):
        MachineReaderAnswer(text="test", span=(1, 5), long_text="long_test", long_text_span=(3, 9),
                            score_reader=0.5, score_answer_in_document=20)


def test_answer_invalid_span():
    with pytest.raises(AssertionError):
        MachineReaderAnswer(text="test", span=(1.5, 2.5), long_text="long_test", long_text_span=(3, 9),
                            score_reader=0.5, score_answer_in_document=0.1)


def test_answer_unclosed_span():
    with pytest.raises(AssertionError):
        MachineReaderAnswer(text="test", span=(1, None), long_text="long_test", long_text_span=(3, 9),
                            score_reader=0.5, score_answer_in_document=0.1)


def test_answer_unopened_span():
    with pytest.raises(AssertionError):
        MachineReaderAnswer(text="test", span=(None, 5), long_text="long_test", long_text_span=(3, 9),
                            score_reader=0.5, score_answer_in_document=0.1)

def test_answer_invalid_long_text_span():
    with pytest.raises(AssertionError):
        MachineReaderAnswer(text="test", span=(1, 5), long_text="long_test", long_text_span=(1.5, 2.5),
                            score_reader=0.5, score_answer_in_document=0.1)


def test_answer_unclosed_long_text_span():
    with pytest.raises(AssertionError):
        MachineReaderAnswer(text="test", span=(1, 5), long_text="long_test", long_text_span=(1, None),
                            score_reader=0.5, score_answer_in_document=0.1)


def test_answer_unopened_long_text_span():
    with pytest.raises(AssertionError):
        MachineReaderAnswer(text="test", span=(1, 5), long_text="long_test", long_text_span=(None, 5),
                            score_reader=0.5, score_answer_in_document=0.1)
