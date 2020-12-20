
# coding=utf-8
from __future__ import absolute_import, division, print_function

import json
import logging
import math
import collections
from io import open
from tqdm import tqdm

from transformers.tokenization_bert import whitespace_tokenize

logger = logging.getLogger(__name__)


class RelationExample(object):
    """
    A single training/test example for the NYT dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 entities=None,
                 answer_ids=None,
                 orig_answer_texts=None,
                 entity_position=None,
                 entity_end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.entities = entities
        self.answer_ids = answer_ids
        self.orig_answer_texts = orig_answer_texts
        self.entity_position = entity_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.entities:
            s += ", entities: %s" % (', '.join(self.entities))
        if self.answer_id:
            s += ", answer_id: %d" % (self.answer_id)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class RelationFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 # tokens,
                 # token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 cls_index,
                 p_mask,
                 paragraph_len,
                 entity_type_ids,
                 candidate_index,
                 candidate_length,
                 index_to_entity_id_map,
                 answer_mask=None,
                 entities=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        # self.tokens = tokens
        # self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_index = cls_index
        self.p_mask = p_mask
        self.paragraph_len = paragraph_len
        self.entity_type_ids = entity_type_ids
        self.candidate_index = candidate_index
        self.candidate_length = candidate_length
        self.index_to_entity_id_map = index_to_entity_id_map
        self.answer_mask = answer_mask
        self.entities = entities
        self.is_impossible = is_impossible


def read_nyt_examples(input_file, is_training, is_with_negative):
    """Read a NYT json file into a list of RelationExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for key, entry in input_data.items():
        document_text = entry["document"]
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        entity_position = {}
        entities = []
        for c in document_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        for p in entry['entities']:
            entity_text = p['text']
            if entity_text == 'NA':
                assert len(entities) == 0
            assert entity_text not in entities
            entities.append(entity_text)
            entity_length = len(entity_text)
            entity_position[entity_text] = []
            for entity_start in p['entity_starts']:
                if entity_start < 0:
                    entity_position[entity_text].append((-1, -1))
                else:
                    t_start = char_to_word_offset[entity_start]
                    t_end = char_to_word_offset[entity_start + entity_length - 1]
                    entity_position[entity_text].append((t_start, t_end))

        assert len(entities) == len(set(entities))
        assert entities[0] == 'NA'
        for qa in entry["qas"]:
            qas_id = qa["id"]
            question_text = qa["question"]
            orig_answer_texts = []
            answer_ids = []
            is_impossible = False
            if is_with_negative:
                is_impossible = qa["is_impossible"]
            if (len(qa["answers"]) < 1) and (not is_impossible):
                raise ValueError(
                    "Each answerable question should have at least 1 answer.")
            if not is_impossible:
                for answer in qa["answers"]:
                    t_orig_answer_text = answer["text"]
                    t_answer_id = entities.index(t_orig_answer_text)
                    assert t_answer_id > 0
                    flag = False
                    answer_length = len(t_orig_answer_text)
                    for answer_offset in answer['answer_starts']:
                        t_start_position = char_to_word_offset[answer_offset]
                        t_end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example. Note that this means for
                        # training mode, every example is NOT guaranteed to be preserved.
                        actual_text = " ".join(doc_tokens[t_start_position:(t_end_position + 1)])
                        cleaned_answer_text = " ".join(whitespace_tokenize(t_orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            logger.warning("Could not find answer: '%s' vs. '%s'",
                                           actual_text, cleaned_answer_text)
                            continue
                        else:
                            flag = True
                            break
                    if flag and t_orig_answer_text:
                        orig_answer_texts.append(t_orig_answer_text)
                        answer_ids.append(t_answer_id)
            else:
                orig_answer_texts.append("NA")
                answer_ids.append(0)

            example = RelationExample(
                qas_id=qas_id,
                question_text=question_text,
                doc_tokens=doc_tokens,
                entities=entities,
                answer_ids=answer_ids,
                orig_answer_texts=orig_answer_texts,
                entity_position=entity_position,
                is_impossible=is_impossible)
            examples.append(example)
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 cls_token_at_end=False, retain_entity=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=0, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000
    # cnt_pos, cnt_neg = 0, 0
    # max_N, max_M = 1024, 1024
    # f = np.zeros((max_N, max_M), dtype=np.float32)

    features = []
    for (example_index, example) in enumerate(tqdm(examples)):

        # if example_index % 100 == 0:
        #     logger.info('Converting %s/%s pos %s neg %s', example_index, len(examples), cnt_pos, cnt_neg)

        query_tokens = tokenizer.tokenize(example.question_text)

        retain_token_indices = set()
        if retain_entity:
            for values in example.entity_position.values():
                for e_s, e_e in values:
                    for i in range(e_s, e_e+1):
                        retain_token_indices.add(i)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            if retain_entity and i in retain_token_indices:
                if tokenizer.basic_tokenizer.do_lower_case:
                    token = token.lower()
                sub_tokens = [token]
            else:
                sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        entity_tok_start_position = []
        entity_tok_end_position = []
        candidate_position = {}

        for key, values in example.entity_position.items():
            for s, e in values:
                if key not in candidate_position:
                    candidate_position[key] = []
                if s < 0:
                    candidate_position[key] = [(-1, -1)]
                    continue
                t_tok_start_position = orig_to_tok_index[s]
                if e < len(example.doc_tokens) - 1:
                    t_tok_end_position = orig_to_tok_index[e + 1] - 1
                else:
                    t_tok_end_position = len(all_doc_tokens) - 1
                entity_tok_start_position.append(t_tok_start_position)
                entity_tok_end_position.append(t_tok_end_position)
                candidate_position[key].append((t_tok_start_position, t_tok_end_position))
        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # Original TF implem also keep the classification token (set to 0) (not sure why...)
            p_mask = []

            # CLS token at the beginning
            if not cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = 0

            # Query
            for i, token in enumerate(query_tokens):
                tokens.append(token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_a_segment_id)
            p_mask.append(1)

            # Paragraph
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(sequence_b_segment_id)
                p_mask.append(0)
            paragraph_len = doc_span.length

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_b_segment_id)
            p_mask.append(1)

            # CLS token at the end
            if cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = len(tokens) - 1  # Index of classification token

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(pad_token)
                input_mask.append(0 if mask_padding_with_zero else 1)
                segment_ids.append(pad_token_segment_id)
                p_mask.append(1)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            # entity_type_ids
            doc_offset = len(query_tokens) + 2
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            entity_type_ids = [0] * len(input_ids)
            for es, ee in zip(entity_tok_start_position, entity_tok_end_position):
                if es >= doc_start and ee <= doc_end:
                    s_p = es - doc_start + doc_offset
                    e_p = ee - doc_start + doc_offset
                    for p in range(s_p, e_p+1):
                        entity_type_ids[p] = 1

            # candidate answer start and end positions
            added_ids = []
            candidate_length = 0
            # according to the document statisitcs, candidate count < 50
            # so we use max candidate _length as 50
            max_candidate_number = 50
            candidate_index = [[0] * max_candidate_number for i in range(2)]
            answer_mask = [0] * max_candidate_number
            index_to_entity_id_map = {}
            entity_id_to_answer_id = {}
            for entity_id, entity in enumerate(example.entities):
                for token_start, token_end in candidate_position[entity]:
                    t_candidate_start = None
                    t_candidate_end = None
                    if token_start < 0:
                        t_candidate_start = 0
                        t_candidate_end = 0
                    elif token_start >= doc_start and token_end <= doc_end:
                        t_candidate_start = token_start - doc_start + doc_offset
                        t_candidate_end = token_end - doc_start + doc_offset
                    if t_candidate_start is not None:
                        candidate_index[0][candidate_length] = t_candidate_start
                        candidate_index[1][candidate_length] = t_candidate_end
                        if entity_id in example.answer_ids:
                            if entity_id in entity_id_to_answer_id:
                                answer_id = entity_id_to_answer_id[entity_id]
                            else:
                                answer_id = len(entity_id_to_answer_id) + 1 # answer_id show larger than zero
                                entity_id_to_answer_id[entity_id] = answer_id
                            answer_mask[candidate_length] = answer_id
                            added_ids.append(entity_id)
                        index_to_entity_id_map[candidate_length] = entity_id
                        candidate_length += 1
            span_is_impossible = example.is_impossible
            if not span_is_impossible:
                if len(added_ids) <= 0:
                    span_is_impossible = True
                    assert sum(answer_mask) == 0
                    answer_mask[0] = 1
                else:
                    assert added_ids[0] != 0

            # make sure the NA is in the first position
            assert candidate_index[0][0] == candidate_index[1][0] == 0
            assert sum(answer_mask) > 0

            if example_index < 10:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                ]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training and span_is_impossible:
                    logger.info("impossible example")

            features.append(
                RelationFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    # tokens=tokens,
                    # token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    cls_index=cls_index,
                    p_mask=p_mask,
                    paragraph_len=paragraph_len,
                    entity_type_ids=entity_type_ids,
                    candidate_index=candidate_index,
                    candidate_length=candidate_length,
                    index_to_entity_id_map=index_to_entity_id_map,
                    answer_mask=answer_mask,
                    entities=example.entities,
                    is_impossible=span_is_impossible))
            unique_id += 1

    return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    new_input_start, new_input_end = [], []
    for text, s, e in zip(orig_answer_text, input_start, input_end):
        tok_answer_text = " ".join(tokenizer.tokenize(text))
        flag = False
        for new_start in range(s, e + 1):
            for new_end in range(e, s - 1, -1):
                text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
                if text_span == tok_answer_text:
                    flag = True
                    new_input_start.append(new_start)
                    new_input_end.append(new_end)
                if flag:
                    break
            if flag:
                break
        if not flag:
            new_input_start.append(s)
            new_input_end.append(e)
    return (new_input_start, new_input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "logits"])


def write_nyt_predictions(all_examples, all_features, all_results,
                          output_prediction_file, output_nbest_file):
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "logit"])

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        nbest = []

        nil_logit = 1000000  # the start logit at the slice with min null score
        seen_prediction = set()
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            for index in range(feature.candidate_length):
                logit = result.logits[index]
                entity_id = feature.index_to_entity_id_map[index]
                entity = feature.entities[entity_id]
                if index:
                    assert entity_id > 0
                    assert entity != 'NA'
                if not entity_id:
                    if logit < nil_logit:
                        nil_logit = logit
                else:
                    start_position = feature.candidate_index[0][index]
                    if not feature.token_is_max_context.get(start_position,
                                                            False):
                        continue
                    seen_prediction.add(entity)
                    nbest.append(
                        _NbestPrediction(
                            text=entity,
                            logit=logit))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if 'NA' not in seen_prediction and '' not in seen_prediction:
            nbest.append(
                    _NbestPrediction(
                        text="",
                        logit=nil_logit))

        if not nbest:
            nbest.append(
                _NbestPrediction(text="", logit=0.))

        assert len(nbest) >= 1

        nbest = sorted(
            nbest,
            key=lambda x: x.logit,
            reverse=True)
        total_na = 0
        for p in nbest:
            if not p.text:
                total_na += 1
        assert total_na > 0
        if total_na > 1:
            print(feature.entities)
        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            text = entry.text
            prob = probs[i]
            '''
            if text in answer_map:
                map_index = answer_map[text]
                alread_prob = nbest_json[map_index]['probability']
                nbest_json[map_index]['probability'] = alread_prob + prob
                continue
            else:
            '''
            output = collections.OrderedDict()
            output["text"] = text
            output["probability"] = prob
            nbest_json.append(output)
        assert len(nbest_json) >= 1

        all_predictions[example.qas_id] = nbest_json[0]["text"]
        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=2) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=2) + "\n")

    return all_predictions


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs
