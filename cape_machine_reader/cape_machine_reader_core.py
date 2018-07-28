from typing import Iterable, Tuple, List, Optional
import numpy as np
from cape_machine_reader.objects.machine_reader_answer import MachineReaderAnswer
from cape_machine_reader.cape_answer_decoder import find_best_spans, softmax


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
        if document_embedding is None:
            doc = self._combine_overlaps(text, before_overlap, after_overlap)
            document_embedding = self.get_document_embedding(doc)

        n_total, n_before, n_text, n_after = map(self._count_tokens, [doc, before_overlap, text, after_overlap])
        assert n_total == (n_before + n_text + n_after), (n_total, n_before, n_text, n_after)
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
        logits_array_start = np.concatenate([
            logits[overlap_start:len(logits) - overlap_end] for (logits, _),  (overlap_start, overlap_end)
            in zip(all_the_logits, all_the_overlaps)
        ])
        logits_array_end = np.concatenate([
            logits[overlap_start:len(logits) - overlap_end] for (_, logits),  (overlap_start, overlap_end)
            in zip(all_the_logits, all_the_overlaps)
        ])

        assert len(logits_array_start) == self._count_tokens(all_combined_texts),\
            'logits length mismatch {} {}'.format(len(logits_array_start), self._count_tokens(all_combined_texts))

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


if __name__ == '__main__':
    import time
    import pickle
    from cape_document_qa import cape_docqa_machine_reader

    cqas = [
        {
            'documents': [
                'The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in '
                'the 10th and 11th centuries gave their name to Normandy, a region in France. They were '
                'descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, '
                'Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles '
                'III of West Francia. Through generations of assimilation and mixing with the native '
                'Frankish and Roman-Gaulish populations, their descendants would gradually merge with the '
                'Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of '
                'the Normans emerged initially in the first half of the 10th century, and it continued to '
                'evolve over the succeeding centuries.',
                'This is another irrelevant document'
            ],
            'question': 'When were the Normans in Normandy?',
            'answer': '10th and 11th centuries'
        },
        {
            'documents': [
                "The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish: Selva Amazónica, "
                "Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch: Amazoneregenwoud), also known in "
                "English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the "
                "Amazon basin of South America. This basin encompasses 7,000,000 square kilometres (2,700,000 sq "
                "mi), of which 5,500,000 square kilometres (2,100,000 sq mi) are covered by the rainforest. This "
                "region includes territory belonging to nine nations. The majority of the forest is contained "
                "within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with "
                "minor amounts in Venezuela, Ecuador, Bolivia, Guyana, Suriname and French Guiana. States or "
                "departments in four nations contain \"\n Amazonas\" in their names. The Amazon represents over "
                "half of the planet's remaining rainforests, and comprises the largest and most biodiverse tract of "
                "tropical rainforest in the world, with an estimated 390 billion individual trees divided into "
                "16,000 species."
            ],
            'question': 'How many square kilometers of rainforest is covered in the basin?',
            'answer': '5,500,000'
        },
        {
            'documents': [
                """The Victoria and Albert Museum (often abbreviated as the V&A), London, is the world's largest 
                museum of decorative arts and design, housing a permanent collection of over 4.5 million objects. It
                was founded in 1852 and named after Queen Victoria and Prince Albert. The V&A is located in the
                Brompton district of the Royal Borough of Kensington and Chelsea, in an area that has become known as
                "Albertopolis" because of its association with Prince Albert, the Albert Memorial and the major 
                cultural institutions with which he was associated. These include the Natural History Museum, the 
                Science Museum and the Royal Albert Hall. The  museum is a non-departmental public body sponsored by 
                the Department for Culture, Media and Sport. Like other national British museums, entrance to the 
                museum has been free since 2001."""
            ],
            'question': 'how many permanent objects are located there?',
            'answer': 'over 4.5 million'
        },
        {
            'documents': [' '.join(['word']*500)],
            'question': 'This is a question to a 500 word document',
            'answer': 'word word word word'
        },
        {
            'documents': [' '.join(['word'] * 1000)],
            'question': 'This is a question to a 1000 word document',
            'answer': 'word word word word'
        }
    ]


    # documents = pickle.load(open(os.path.join(DATA_FOLDER, 'documents.pickle'), 'rb'))
    # num_words = 10000
    # cqas = [
    #     {
    #         'documents': [' '.join(documents[0].split()[:num_words])],
    #         'question': 'how many permanent objects are located in the Victoria and Albert Museum?',
    #         'answer': 'over 4.5 million objects'
    #     },
    #     {
    #         'documents': [' '.join(documents[1].split()[:num_words])],
    #         'question': 'how many permanent objects are located in the Victoria and Albert Museum?',
    #         'answer': 'over 4.5 million objects'
    #     }
    # ]
    #model = cape_docqa_machine_reader.CapeDocQAMachineReaderModel(
    model = cape_docqa_machine_reader.RandomMachineReaderModel(
        cape_docqa_machine_reader.get_production_model_config('../../document-qa/production_ready_model')
    )

    configuration = MachineReaderConfiguration(threshold_reader=0,
                                               threshold_answer_in_document=0, top_k=1)
    machine_reader = MachineReader(model)

    VERBOSE_ON_ERROR = True
    nb_correct = 0
    for cqa in cqas:
        t0 = time.time()
        for document in cqa['documents']:
            print("Number of words in document: {}".format(len(document.split())))
            for answer in machine_reader.get_answers(configuration, document, cqa['question']):
                nb_correct += answer.text == cqa['answer']
                if True or VERBOSE_ON_ERROR and answer.text != cqa['answer']:
                    # print('document containing answer  :', document)
                    print('question                    :', cqa['question'])
                    print('gold answer                 :', cqa['answer'])
                    print('predicted answer            :', answer.text)
                    print('predicted answer span       :', answer.span)
                    print('score_reader                :', answer.score_reader)
                    print('score_answer_in_document    :', answer.score_answer_in_document)
                    print('answer long text            :', answer.long_text)
                    print('answer long text span       :', answer.long_text_span)
                    print('words in all documents      :', len(' '.join(cqa['documents']).split()))
                    print('time to predict             :', time.time() - t0)
                    print()

    if VERBOSE_ON_ERROR and nb_correct < len(cqas):
        print('{}/{} correct predictions'.format(nb_correct, len(cqas)))
    print(float(nb_correct))

    print()
    print('-' * 80)
    print()
