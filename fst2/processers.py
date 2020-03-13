import copy
import os
import csv
import json
import torch
import logging
from transformers.file_utils import is_tf_available, is_torch_available
from functools import wraps
from .utils import CACHE_PARAMS


logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensorflow tensors
        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def tfds_map(self, example):
        """Some tensorflow_datasets datasets are not formatted the same way the GLUE datasets are.
        This method converts examples to the correct format."""
        if len(self.get_labels()) > 1:
            example.label = self.get_labels()[int(example.label)]
        return example

    @classmethod
    def _read_csv(cls, input_file, delimiter="\t", quotechar=None):
        """Reads a tab separated csv/tsv file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter=delimiter, quotechar=quotechar))


class SingleSentenceClassificationProcessor(DataProcessor):
    """ Generic processor for a single sentence classification data set."""

    def __init__(self, labels=None, examples=None, mode="classification", verbose=False):
        self.labels = [] if labels is None else labels
        self.examples = [] if examples is None else examples
        self.mode = mode
        self.verbose = verbose

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return SingleSentenceClassificationProcessor(labels=self.labels, examples=self.examples[idx])
        return self.examples[idx]

    @classmethod
    def create_from_csv(
        cls, file_name, delimiter, split_name="", column_label=0, column_text=1, column_id=None, skip_first_row=False, **kwargs
    ):
        processor = cls(**kwargs)
        processor.add_examples_from_csv(
            file_name,
            delimiter,
            split_name=split_name,
            column_label=column_label,
            column_text=column_text,
            column_id=column_id,
            skip_first_row=skip_first_row,
            overwrite_labels=True,
            overwrite_examples=True,
        )
        return processor

    @classmethod
    def create_from_examples(cls, texts_or_text_and_labels, labels=None, **kwargs):
        processor = cls(**kwargs)
        processor.add_examples(texts_or_text_and_labels, labels=labels)
        return processor

    def add_examples_from_csv(
        self,
        file_name,
        delimiter,
        split_name="",
        column_label=0,
        column_text=1,
        column_id=None,
        skip_first_row=False,
        overwrite_labels=False,
        overwrite_examples=False,
    ):
        lines = self._read_csv(file_name, delimiter=delimiter)
        if skip_first_row:
            lines = lines[1:]
        texts = []
        labels = []
        ids = []
        for (i, line) in enumerate(lines):
            texts.append(line[column_text])
            labels.append(line[column_label])
            if column_id is not None:
                ids.append(line[column_id])
            else:
                guid = "%s-%s" % (split_name, i) if split_name else "%s" % i
                ids.append(guid)

        return self.add_examples(
            texts, labels, ids, overwrite_labels=overwrite_labels, overwrite_examples=overwrite_examples
        )

    def add_examples(
        self, texts_or_text_and_labels, labels=None, ids=None, overwrite_labels=False, overwrite_examples=False
    ):
        assert labels is None or len(texts_or_text_and_labels) == len(labels)
        assert ids is None or len(texts_or_text_and_labels) == len(ids)
        if ids is None:
            ids = [None] * len(texts_or_text_and_labels)
        if labels is None:
            labels = [None] * len(texts_or_text_and_labels)
        examples = []
        added_labels = set()
        for (text_or_text_and_label, label, guid) in zip(texts_or_text_and_labels, labels, ids):
            if isinstance(text_or_text_and_label, (tuple, list)) and label is None:
                text, label = text_or_text_and_label
            else:
                text = text_or_text_and_label
            added_labels.add(label)
            examples.append(InputExample(guid=guid, text_a=text, text_b=None, label=label))

        # Update examples
        if overwrite_examples:
            self.examples = examples
        else:
            self.examples.extend(examples)

        # Update labels
        if overwrite_labels:
            self.labels = list(added_labels)
        else:
            self.labels = list(set(self.labels).union(added_labels))

        return self.examples

    def get_features(
        self,
        tokenizer,
        max_length=None,
        pad_on_left=False,
        pad_token=0,
        mask_padding_with_zero=True,
        return_tensors="pt",
    ):
        """
        Convert examples in a list of ``InputFeatures``

        Args:
            tokenizer: Instance of a tokenizer that will tokenize the examples
            max_length: Maximum example length
            task: GLUE task
            label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
            output_mode: String indicating the output mode. Either ``regression`` or ``classification``
            pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
            pad_token: Padding token
            mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
                and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
                actual values)

        Returns:
            If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
            containing the task-specific features. If the input is a list of ``InputExamples``, will return
            a list of task-specific ``InputFeatures`` which can be fed to the model.

        """
        if max_length is None:
            max_length = tokenizer.max_len

        label_map = {label: i for i, label in enumerate(self.labels)}

        all_input_ids = []
        for (ex_index, example) in enumerate(self.examples):
            if ex_index % 10000 == 0:
                logger.info("Tokenizing example %d", ex_index)

            input_ids = tokenizer.encode(
                example.text_a, add_special_tokens=True, max_length=min(max_length, tokenizer.max_len),
            )
            all_input_ids.append(input_ids)

        batch_length = max(len(input_ids) for input_ids in all_input_ids)

        features = []
        for (ex_index, (input_ids, example)) in enumerate(zip(all_input_ids, self.examples)):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d/%d" % (ex_index, len(self.examples)))
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = batch_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)

            assert len(input_ids) == batch_length, "Error with input length {} vs {}".format(
                len(input_ids), batch_length
            )
            assert len(attention_mask) == batch_length, "Error with input length {} vs {}".format(
                len(attention_mask), batch_length
            )

            if self.mode == "classification":
                label = label_map[example.label]
            elif self.mode == "regression":
                label = float(example.label)
            else:
                raise ValueError(self.mode)

            if ex_index < 5 and self.verbose:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info("label: %s (id = %d)" % (example.label, label))

            features.append(InputFeatures(input_ids=input_ids, attention_mask=attention_mask, label=label))

        if return_tensors is None:
            return features
        elif return_tensors == "tf":
            if not is_tf_available():
                raise RuntimeError("return_tensors set to 'tf' but TensorFlow 2.0 can't be imported")
            import tensorflow as tf

            def gen():
                for ex in features:
                    yield ({"input_ids": ex.input_ids, "attention_mask": ex.attention_mask}, ex.label)

            dataset = tf.data.Dataset.from_generator(
                gen,
                ({"input_ids": tf.int32, "attention_mask": tf.int32}, tf.int64),
                ({"input_ids": tf.TensorShape([None]), "attention_mask": tf.TensorShape([None])}, tf.TensorShape([])),
            )
            return dataset
        elif return_tensors == "pt":
            if not is_torch_available():
                raise RuntimeError("return_tensors set to 'pt' but PyTorch can't be imported")
            import torch
            from torch.utils.data import TensorDataset

            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            if self.mode == "classification":
                all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
            elif self.mode == "regression":
                all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

            dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels)
            return dataset
        else:
            raise ValueError("return_tensors should be one of 'tf' or 'pt'")


class SequnceTokenClassificationProcessor(DataProcessor):
    def __init__(self, labels=None, examples=None, mode="classification", verbose=False):
        self.labels = [] if labels is None else labels
        self.examples = [] if examples is None else examples
        self.mode = mode
        self.verbose = verbose
    
    @classmethod
    def create_from_txt(cls, file_name, delimiter, **kwargs): 
        processor = cls(**kwargs)
        processor.read_examples_from_txt(file_name, delimiter)
        return processor
        
    def read_examples_from_txt(self, file_name, delimiter):
        examples = []
        guid_index = 1
        with open(file_name, encoding="utf-8") as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        examples.append(InputExample(guid=guid_index,
                                                     text_a=words,
                                                     label=labels))
                        words = []
                        labels = []
                        guid_index +=1
                else:
                    splits = line.split(delimiter)  
                    if len(splits) > 1:
                        words.append(splits[0])
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        labels.append("O")
            if words:
                examples.append(InputExample(guid=guid_index, text_a=words,
                                              label=labels))
        self.examples = examples
       
    def get_features(
        self,
        max_seq_length,
        tokenizer,
        return_tensors,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=0,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
    ):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
        label_map = {label: i for i, label in enumerate(self.labels)}

        features = []  # [[inout_ids, input_mask, segment_ids, label_id]]
        for (ex_index, example) in enumerate(self.examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d of %d", ex_index, len(self.examples))

            tokens = []
            label_ids = []
            for word, label in zip(example.text_a, example.label):
                word_tokens = tokenizer.tokenize(word)  
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))  # some language like german one word will be tokenized to several subwords

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens) > max_seq_length - special_tokens_count:  
                tokens = tokens[: (max_seq_length - special_tokens_count)]
                label_ids = label_ids[: (max_seq_length - special_tokens_count)]
            
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
                label_ids += [pad_token_label_id]
            segment_ids = [pad_token_segment_id] * len(tokens)

            if cls_token_at_end:
                tokens += [cls_token]
                label_ids += [pad_token_label_id]
                segment_ids += [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                label_ids = [pad_token_label_id] + label_ids
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)  # 输入的是列表， 返回的也是列表

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                label_ids = ([pad_token_label_id] * padding_length) + label_ids
            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                label_ids += [pad_token_label_id] * padding_length
            # 保证我的input_id, input_mask, segment_ids, 和label_ids都是一致的
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s", example.guid)
                logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
                logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
            
            features.append(
                InputFeatures(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, label=label_ids)
            )  # note segement_ids has 1 in the front? but we don't use it right now

        # fetures to dataset 
        if return_tensors is None:
            return features    
        elif return_tensors == "tf":
            if not is_tf_available():
                raise RuntimeError("return_tensors set to 'tf' but TensorFlow 2.0 can't be imported")
            import tensorflow as tf

            def gen():
                for ex in features:
                    yield ({"input_ids": ex.input_ids, "attention_mask": ex.attention_mask}, ex.label)

            dataset = tf.data.Dataset.from_generator(
                gen,
                ({"input_ids": tf.int32, "attention_mask": tf.int32}, tf.int64),
                ({"input_ids": tf.TensorShape([None]), "attention_mask": tf.TensorShape([None])}, tf.TensorShape([])),
            )
            return dataset
        elif return_tensors == "pt":
            if not is_torch_available():
                raise RuntimeError("return_tensors set to 'pt' but PyTorch can't be imported")
            import torch
            from torch.utils.data import TensorDataset

            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
            if self.mode == "classification":
                all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
            elif self.mode == "regression":
                all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
            return dataset
        else:
            raise ValueError("return_tensors should be one of 'tf' or 'pt'")       


##############   help functions ######
def load_and_cache_dataset(func):
    @wraps(func)
    def inner(*args, **kwargs):
        logger = logging.getLogger("Load-Cache_Dataset")
        cached_features_file = os.path.join(
            CACHE_PARAMS["data_dir"],
            "cached_{}_{}_{}".format(
                CACHE_PARAMS["mode"], list(filter(None, CACHE_PARAMS["model_name_or_path"].split("/"))).pop(), 
                str(CACHE_PARAMS["max_seq_length"])
            ),)
        if os.path.exists(cached_features_file):
            dataset = torch.load(cached_features_file)
            logger.info("Load dataset from {}".format(cached_features_file))
            return dataset
        else:
            logger.info("Read data and preparing dataset")
            dataset = func(*args, **kwargs)
            torch.save(dataset, cached_features_file)
            logger.info("Cached dataset at {}".format(cached_features_file))
            return dataset
    return inner



# if __name__ == "__main__":
#     from transformers import BertTokenizer
#     processer = SingleSentenceClassificationProcessor.create_from_csv(file_name="test_data.tsv", delimiter=",")
#     tokenizer = BertTokenizer.from_pretrained("/Users/jiangchaodi/Code/NLP/transformer_examples/models/chinese_L-12_H-768_A-12")
#     dataset = processer.get_features(tokenizer=tokenizer, max_length=128)
#     print(dataset)

#     test torch.load(dataset)
#     import torch
#     torch.save(dataset, "test_cached")
#     dataset2 = torch.load("test_cached")
#     print(dataset2)

# if __name__ == "__main__":
#     from transformers import BertTokenizer
#     from pipelines import get_labels
#     processer = SequnceTokenClassificationProcessor.create_from_txt(file_name="/Users/jiangchaodi/Code/NLP/fasttransformer/fst2/data/train.txt", delimiter="\t", labels=get_labels("/Users/jiangchaodi/Code/NLP/fasttransformer/fst2/data/labels.txt"))
#     tokenizer = BertTokenizer.from_pretrained("/Users/jiangchaodi/Code/NLP/transformer_examples/models/chinese_L-12_H-768_A-12")
#     dataset = processer.get_features(tokenizer=tokenizer, max_seq_length=128, return_tensors="pt")
#     print(dataset)

