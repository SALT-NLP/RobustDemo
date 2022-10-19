import os
import random
import string
import torch
import logging as log

from torch.utils.data import TensorDataset


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels, data_dir=None):
        """Constructs a InputExample.
		Args:
			guid: Unique id for the example.
			words: list. The words of the sequence.
			labels: (Optional) list. The labels for each word of the sequence. This should be
			specified for train and dev examples, but not for test examples.
		"""
        self.guid = guid
        self.words = words
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, label_ids, subtoken_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_ids = label_ids
        self.subtoken_ids = subtoken_ids


def get_labels(task, dataset, tagging="BIO"):
    if task == 'ner':
        if dataset == 'conll2003':
            if tagging == "IO":
                return ["O", "I-LOC", "I-MISC", "I-ORG", "I-PER"]
            else:
                return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
        elif dataset == 'ontonotes':
            if tagging == "IO":
                return ["O", "I-EVENT", "I-FAC", "I-GPE", "I-LANGUAGE", "I-LAW", "I-LOC", "I-NORP",
                        "I-ORG", "I-PERSON", "I-PRODUCT", "I-WORK_OF_ART"]
            else:
                return ["O", "B-PERSON", "I-PERSON", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-EVENT", "I-EVENT",
                        "B-LANGUAGE", "I-LANGUAGE", "B-LAW", "I-LAW", "B-PRODUCT", "I-PRODUCT",
                        "B-FAC", "I-FAC", "B-WORK_OF_ART", "I-WORK_OF_ART", "B-NORP", "I-NORP", "B-GPE", "I-GPE"]
        elif dataset in ('NRB', 'WTS'):
            if tagging == "IO":
                return ["O", "I-LOCATION", "I-MISC", "I-ORGANIZATION", "I-PERSON"]
            else:
                raise ValueError("NRB_WTS can only use IO tagging")
        else:
            raise ValueError("{} dataset is not supported, only conll2003 and ontonotes are available".format(dataset))
    else:
        if dataset == 'conll2000':
            if tagging == "IO":
                return ['O', 'I-ADJP', 'I-ADVP', 'I-NP', 'I-PP', 'I-SBAR', 'I-VP']
            else:
                return ['O', 'B-ADJP', 'I-ADJP', 'B-ADVP', 'I-ADVP', 'B-NP', 'I-NP', 'B-PP', 'I-PP', 'B-SBAR', 'I-SBAR', 'B-VP', 'I-VP']
        else:
            raise ValueError("{} dataset is not supported, only conll2003 and ontonotes are available".format(dataset))


def read_data_from_file(file_path, mode, label_map, tagging):
    guid_index = 0
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(InputExample(
                        guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split()
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                    if tagging == "IO" and labels[-1].startswith("B"):
                        labels[-1] = "I" + labels[-1][1:]
                    if labels[-1] not in label_map:
                        labels[-1] = 'O'
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(InputExample(
                guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
    return examples


def convert_examples_to_features(mode, demonstration, examples, label_list,
                                 max_seq_length, tokenizer, logger,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 pad_token_label_id=-100, mask_padding_with_zero=True):
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    max_len = 0

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0 and logger is not None:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        tokens = []
        label_ids = []
        subtoken_ids = []
        # this subtoken_ids array is used to mark whether the token is a subtoken of a word or not
        for word, label in zip(example.words, example.labels):

            word_tokens = tokenizer.tokenize(' ' + word)
            tokens.extend(word_tokens)

            if len(word_tokens) > 0:
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
                subtoken_ids.extend([1] + [0] * (len(word_tokens) - 1))

        # if mode == 'replicate':
        #     demonstration = sep_token + ' ' + ' '.join(example.words)
        # elif mode == 'replicate_4':
        #     demonstration = (sep_token + ' ' + ' '.join(example.words) + ' ') * 4
        for word in demonstration.split():
            if mode.startswith('random_'):
                word_tokens = [word]
            else:
                word_tokens = tokenizer.tokenize(' ' + word)

            tokens.extend(word_tokens)
            if len(word_tokens) > 0:
                label_ids.extend([pad_token_label_id] * len(word_tokens))
                subtoken_ids.extend([-1] * (len(word_tokens)))
                # subtoken_ids.extend([1] + [0] * (len(word_tokens) - 1))

        if len(tokens) > max_len:
            max_len = len(tokens)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]
            label_ids = label_ids[: (max_seq_length - 2)]
            subtoken_ids = subtoken_ids[:(max_seq_length - 2)]

        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        subtoken_ids += [-1]

        tokens = [cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids
        subtoken_ids = [-1] + subtoken_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        label_ids += [pad_token_label_id] * padding_length
        subtoken_ids += [-1] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(subtoken_ids) == max_seq_length

        if ex_index < 2 and logger is not None:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
            logger.info("subtoken_ids: %s", " ".join([str(x) for x in subtoken_ids]))

        features.append(
            InputFeatures(input_ids=input_ids, input_mask=input_mask, label_ids=label_ids, subtoken_ids=subtoken_ids)
        )
    if logger is not None:
        logger.info('=*' * 40)
        logger.info('max_len: {}'.format(max_len))
    return features


def read_data(args, tokenizer, logger, mode):
    labels = get_labels(args.task, args.dataset, args.tagging)
    if args.dataset == "NRB":
        if mode != 'test':
            raise ValueError("NRB dataset is only for evaluate!")
        file_path = os.path.join('data', args.task, "NRB_WTS", "en.nrb.conll")
    elif args.dataset == "WTS":
        if mode != 'test':
            raise ValueError("WTS dataset is only for evaluate!")
        file_path = os.path.join('data', args.task, "NRB_WTS", "en.wts.conll")
    elif args.few_shot_dir is not None and mode == 'train':
        file_path = os.path.join(args.few_shot_dir, "{}.txt".format(mode))
    else:
        file_path = os.path.join('data', args.task, args.dataset, "{}.txt".format(mode))
    logger.info("Preprocessing data from dataset file at %s.", file_path)
    examples = read_data_from_file(file_path, mode, labels, args.tagging)
    if args.mode in ('no', 'replicate', 'replicate_4'):
        demonstration = ''
    else:
        demonstration_path = os.path.join(args.few_shot_dir, 'demonstration_' + args.mode + ('_roberta' if args.model_type == 'roberta' else '') + '.txt')
        with open(demonstration_path, 'r') as f:
            demonstration = f.readline()
    print('demonstration:', demonstration)
    print('data num: {}'.format(len(examples)))

    features = convert_examples_to_features(args.mode, demonstration,
                                            examples, labels, args.max_seq_length, tokenizer, logger,
                                            cls_token=tokenizer.cls_token, sep_token=tokenizer.sep_token,
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0], )

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_subtoken_ids = torch.tensor([f.subtoken_ids for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids, all_subtoken_ids)

    return dataset
