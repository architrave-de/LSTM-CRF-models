import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from tqdm import tqdm
from bionlp.data.token import Token
from bionlp.data.sentence import Sentence
from bionlp.data.document import Document
from bionlp.data.dataset import Dataset


def encode_data_format(documents, raw_text, umls_params, sentence_limit=0, label_blacklist=[]):
    logger.info('Encoding dataset into data format')
    document_dict = {did_: (dtext_, metamap_)
                     for (did_, dtext_, metamap_) in raw_text}
    documentList = []
    total_sent_count = 0
    sent_limit_reached = False
    for did, document in tqdm(documents):
        sentenceList = []
        sid = 0
        for sent in document:
            sent_is_blacklisted = False
            tid = 0
            tokenList = []
            for token in sent:
                label = token[2]
                if label in label_blacklist:
                    sent_is_blacklisted = True
                    break
                else:
                    tokenList.append(Token(value=token[0], id=tid,
                        document=did, offset=token[1], Annotation=label))
                    tid += 1
            if not sent_is_blacklisted:
                newSentence = Sentence(tokenList, sid)
                sentenceList.append(newSentence)
                sid += 1
                total_sent_count += 1
                if sentence_limit != 0 and total_sent_count >= sentence_limit:
                    sent_limit_reached = True
                    break
        newDocument = Document(sentenceList, did)
        newDocument.attr['raw_text'] = document_dict[did][0]
        if umls_params != 0:
            newDocument.attr['metamap_anns'] = document_dict[did][1]
        documentList.append(newDocument)
        if sent_limit_reached:
            break
    dataset = Dataset(documentList)
    dataset.passive.append('Annotation')
    return dataset


def decode_training_data(dataset):
    documentList = []
    for document in dataset.value:
        sentenceList = []
        for sent in document.value:
            sentenceList.append(
                [((token.value, token), token.attr['Annotation']) for token in sent.value])
        documentList.append(sentenceList)
    logger.info('Number of Records decoded into training data format {0}'.format(
        documentList.__len__()))
    return documentList


def decode_n_strip_training_data(dataset):
    # Used for decoding the training data if required for Neural Network
    # Models.
    documentList = []
    for document in dataset.value:
        sentenceList = []
        for sent in document.value:
            sentenceList.append(
                [(token.value, token.attr['Annotation']) for token in sent.value])
        documentList.append(sentenceList)
    logger.info('Number of Records decoded into training data format {0}'.format(
        documentList.__len__()))
    return documentList
