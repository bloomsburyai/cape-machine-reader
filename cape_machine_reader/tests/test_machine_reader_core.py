from cape_machine_reader.tests.test_machine_reader_model import DummyMachineReaderModel
from cape_machine_reader.cape_machine_reader_core import MachineReader, MachineReaderConfiguration
from pytest import fixture


@fixture
def dummy_machine_reader_model():
    return DummyMachineReaderModel()


@fixture
def dummy_mr_config():
    return MachineReaderConfiguration()


@fixture
def context():
    return '''"Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50."'''


@fixture
def question():
    return "Which NFL team represented the AFC at Super Bowl 50?"

@fixture
def before_text():
    return 'This is some before text. '


@fixture
def after_text():
    return ' This is some text to go afterwards.'


def test_get_answers_produces_answers(dummy_machine_reader_model, dummy_mr_config, context, question):
    mr = MachineReader(dummy_machine_reader_model)
    answers = mr.get_answers(dummy_mr_config, context, question)
    n_answers = len([a for a in answers])
    assert n_answers > 0


def test_get_answers_produces_correct_n_answers(dummy_machine_reader_model, dummy_mr_config, context, question):
    dummy_mr_config.top_k = 10
    mr = MachineReader(dummy_machine_reader_model)
    answers = mr.get_answers(dummy_mr_config, context, question)
    n_answers = len([a for a in answers])
    assert n_answers == dummy_mr_config.top_k


def test_get_answers_from_logits_correct_n_answers(dummy_machine_reader_model, dummy_mr_config, context, question):
    dummy_mr_config.top_k = 10
    mr = MachineReader(dummy_machine_reader_model)
    logits, overlaps = mr.get_logits(context, question)
    answers = mr.get_answers_from_logits(
        dummy_mr_config, [logits], [overlaps], context)
    n_answers = len([a for a in answers])
    assert n_answers == dummy_mr_config.top_k


def test_document_embedding_correct_shape(dummy_machine_reader_model, context, before_text, after_text):
    mr = MachineReader(dummy_machine_reader_model)
    doc_emb = mr.get_document_embedding(context, before_overlap=before_text,after_overlap=after_text)
    assert doc_emb.shape[0] == len(mr.model.tokenize(context)[0]) + len(mr.model.tokenize(before_text)[0]) + len(mr.model.tokenize(after_text)[0])


def test_get_logits_correct_shape_no_doc_emb(dummy_machine_reader_model, dummy_mr_config, context, question, before_text, after_text):
    mr = MachineReader(dummy_machine_reader_model)
    (start_logits, end_logits), (n_bef, n_aft) = mr.get_logits(context, question, before_overlap=before_text, after_overlap=after_text)
    len_c = len(mr.model.tokenize(context)[0])
    len_b = len(mr.model.tokenize(before_text)[0])
    len_a = len(mr.model.tokenize(after_text)[0])
    expected_shape = len_c + len_b + len_a
    assert len(start_logits) == expected_shape
    assert len(end_logits) == expected_shape
    assert n_bef == len_b
    assert n_aft == len_a


def test_get_logits_correct_shape_with_doc_emb(dummy_machine_reader_model, dummy_mr_config, context, question, before_text, after_text):
    mr = MachineReader(dummy_machine_reader_model)
    doc_emb = mr.get_document_embedding(context, before_overlap=before_text, after_overlap=after_text)
    (start_logits, end_logits), (n_bef, n_aft) = mr.get_logits(context, question, before_overlap=before_text, after_overlap=after_text, document_embedding=doc_emb)
    len_c = len(mr.model.tokenize(context)[0])
    len_b = len(mr.model.tokenize(before_text)[0])
    len_a = len(mr.model.tokenize(after_text)[0])
    expected_shape = len_c + len_b + len_a
    assert len(start_logits) == expected_shape
    assert len(end_logits) == expected_shape
    assert n_bef == len_b
    assert n_aft == len_a

