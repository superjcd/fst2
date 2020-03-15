import os
import logging
import torch
from typing import Dict
from abc import ABC, abstractmethod
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, DistributedSampler
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForTokenClassification,
    BertForSequenceClassification, 
    BertTokenizer,
    CamembertConfig,
    CamembertForTokenClassification,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForTokenClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizer,
    XLMRobertaConfig,
    XLMRobertaForTokenClassification,
    XLMRobertaTokenizer,
    get_linear_schedule_with_warmup,
)
from .processers import (
    load_and_cache_dataset, 
    SingleSentenceClassificationProcessor,
    SequnceTokenClassificationProcessor
    )
from .utils import (
    CACHE_PARAMS, 
    set_seed, 
    get_tokenclassification_labels, 
    get_textclassification_labels, 
    get_label2id, 
    get_id2label)
from .commons import _train, _evaluate, _predict


# prepare logging
logging.basicConfig(level = logging.INFO,
                     format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     datefmt="%m/%d/%Y %H:%M:%S")


# all supported nlp pipeline

# here is just one task:ner, add  other task
MODEL_CLASSES = {
    "ner":{
        "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
        "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer),
        "distilbert": (DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer),
        "camembert": (CamembertConfig, CamembertForTokenClassification, CamembertTokenizer),
        "xlmroberta": (XLMRobertaConfig, XLMRobertaForTokenClassification, XLMRobertaTokenizer),
          },
    "text-classification":{
        "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    }
}


def pipeline(configs: Dict):
    """
       factory function to chose a Pipeline for trainning given a specific task name
    """
    # configs[task]
    try:
        nlp = SUPPORTED_PIPELINE[configs["pipeline"]["task"]]
    except KeyError as e:
        print(e.args)
        raise e
    return nlp(configs)


####################  Pipeline  Abstract Class #################
class Pipeline(ABC):
    """
       Base Interface for trainning NLP model
    """
    @abstractmethod
    def get_data(self, *args, **kwargs):
        """
           get dataset for nlp trainning
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def set_attrs(self, configs: Dict):
        """
          Set property for instance recursively

          Args:
            configs: dict
          
          Returns:
            None
        """
        for k, v in configs.items():
            if isinstance(v, dict):
                self.set_attrs(v)  # if value is a dict call the function again
            else:
                setattr(self, k, v)

    def prepare(self):
        if self.local_rank in [-1, 0]:  # -1 是本地
            self.tb_writer = SummaryWriter(self.tensorboard_dir)
        # Setup CUDA, GPU & distributed training, TODO: now is defaultly support gpu and distrubiute training, later on we can mannuly do this
        if self.local_rank == -1 or self.no_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
            self.n_gpu = torch.cuda.device_count()
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            self.n_gpu = 1


        # Load pretrained model and tokenizer
        if self.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        if self.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
        
        # CHANGE CACHE logging params
        CACHE_PARAMS["data_dir"] = self.data_dir
        CACHE_PARAMS["model_name_or_path"] = self.model_name_or_path
        CACHE_PARAMS["max_seq_length"] = self.max_seq_length



#####################  Task Specfic Mixin #####################
class NerMixin():  # or just class
    # prepare data
    def prepare_all(self):
        super(NerMixin, self).prepare()  # call Pipeline prepare
        self.labels = get_tokenclassification_labels(self.label_file)
        self.num_labels = len(self.labels)
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = CrossEntropyLoss().ignore_index  # CHECK: ignore_index -> pad的index， 不计入计算的

        self.config_class, self.model_class, self.tokenizer_class = \
            MODEL_CLASSES[self.task.lower()][self.model_type.lower()]

        self.config = self.config_class.from_pretrained(
            self.config_name if self.config_name else self.model_name_or_path,
            num_labels=self.num_labels, label2id=get_label2id(self.labels),
            id2label=get_id2label(self.labels)
        )
        self.tokenizer = self.tokenizer_class.from_pretrained(
            self.tokenizer_name if self.tokenizer_name else self.model_name_or_path,
            do_lower_case=self.do_lower_case
        )
        self.model = self.model_class.from_pretrained(  # 加载模型的时候需要config
            self.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_name_or_path),
            config=self.config
        )
        self.model.to(self.device)

class TextClassfictionMixin():  # or just class
    # prepare data
    def prepare_all(self):
        super().prepare() # call Pipeline prepare
        self.labels = get_textclassification_labels(self.label_file)
        self.num_labels = len(self.labels)
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = CrossEntropyLoss().ignore_index  # CHECK: ignore_index -> pad的index， 不计入计算的

        self.config_class, self.model_class, self.tokenizer_class = \
            MODEL_CLASSES[self.task.lower()][self.model_type.lower()]

        self.config = self.config_class.from_pretrained(
            self.config_name if self.config_name else self.model_name_or_path,
            num_labels=self.num_labels, label2id=get_label2id(self.labels),
            id2label=get_id2label(self.labels)
        )
        self.tokenizer = self.tokenizer_class.from_pretrained(
            self.tokenizer_name if self.tokenizer_name else self.model_name_or_path,
            do_lower_case=self.do_lower_case
        )
        self.model = self.model_class.from_pretrained(  # 加载模型的时候需要config
            self.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_name_or_path),
            config=self.config
        )
        self.model.to(self.device)


#################### NLP Pipelines  ##########################

###### NER ######
class NerPipeline(NerMixin, Pipeline):
    """
        pipeline for sequence token classification.
    """
    def __init__(self, configs):
        self.configs = configs
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Training/evaluation parameters %s", self.configs)
        self.set_attrs(configs)  # set attributes for instance

    @load_and_cache_dataset  
    def get_data(self, **kwargs):
        mode = CACHE_PARAMS["mode"]    # mode is for naming the cache dataset, e.g when "mode" is set to "train" then *_train_* under data directory.
        data_file = os.path.join(self.data_dir, "{}.txt".format(mode))
        processor = SequnceTokenClassificationProcessor.create_from_txt(file_name=data_file, labels=self.labels, **kwargs)
        return processor.get_features(tokenizer=self.tokenizer, max_seq_length=self.max_seq_length, return_tensors="pt") 

    def train(self):
        CACHE_PARAMS["mode"] = "train" 
        dataset = self.get_data(delimiter=self.delimiter)
        _train(
            task=self.task,
            logger=self.logger, tb_writer=self.tb_writer, model=self.model,
            tokenizer=self.tokenizer, dataset=dataset, max_steps=self.max_steps,
            num_train_epochs=self.num_train_epochs, gradient_accumulation_steps=self.gradient_accumulation_steps, 
            weight_decay=self.weight_decay, learning_rate=self.learning_rate,
            adam_epsilon=self.adam_epsilon, max_grad_norm=self.max_grad_norm,
            warmup_steps=self.warmup_steps,
            fp16=self.fp16, fp16_opt_level=self.fp16_opt_level,
            n_gpu=self.n_gpu,
            local_rank=self.local_rank,
            evaluate_during_training=self.evaluate_during_training,
            evaluate_func=self.evaluate,
            per_gpu_train_batch_size=self.per_gpu_train_batch_size,
            device=self.device,
            output_dir=self.output_dir, model_type=self.model_type,
            model_name_or_path=self.model_name_or_path,
            configs=self.configs, seed=self.seed,
            logging_steps=self.logging_steps, save_steps=self.save_steps
            )
        
    def evaluate(self, mode, model, tokenizer):
        """
        """
        CACHE_PARAMS["mode"] = mode  # evaluate can be used as evaluation for trainning, or prediction, so need to be set depended on demands 
        eval_dataset = self.get_data(delimiter=self.delimiter)
        return _evaluate(task=self.task, logger=self.logger, 
                   model_type=self.model_type, tokenizer=tokenizer, model=model,
                   dataset=eval_dataset, labels=self.labels, pad_token_label_id=self.pad_token_label_id,
                   per_gpu_eval_batch_size=self.per_gpu_eval_batch_size, n_gpu=self.n_gpu,
                   local_rank=self.local_rank, device=self.device, ouput_index=2)

    def predict(self):
        CACHE_PARAMS["mode"] = "test"
        _predict(task=self.task,
                    evaluate_func=self.evaluate,
                    logger=self.logger,
                    result_dir=self.result_dir,
                    data_dir=self.data_dir,
                    output_dir=self.output_dir,
                    test_file_name="test.txt",
                    delimiter=self.delimiter,
                    column_text=0,
                    do_lower_case=self.do_lower_case,
                    tokenizer_class=self.tokenizer_class,
                    model_class=self.model_class, 
                    device=self.device,
                    prediction_model_dir=self.prediction_model_dir)

    def __call__(self):
        """
        """
        self.prepare_all()
        if self.do_train:
            self.train()
        if self.do_predict and self.local_rank in [-1, 0]:
            self.predict()


#################        TextClassification Pipeline   ###########
class TextClassificationPipeline(TextClassfictionMixin, Pipeline):
    def __init__(self, configs):
        self.configs = configs
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Training/evaluation parameters %s", self.configs)
        self.set_attrs(configs)  # set attributes for instance


    @load_and_cache_dataset
    def get_data(self, **kwargs):
        mode = CACHE_PARAMS["mode"]
        data_file = os.path.join(self.data_dir, "{}.csv".format(mode))
        processor = SingleSentenceClassificationProcessor.create_from_csv(data_file, **kwargs)
        return processor.get_features(tokenizer=self.tokenizer, max_length=self.max_seq_length)

    def train(self): 
        CACHE_PARAMS["mode"] = "train" 
        dataset = self.get_data(delimiter=self.delimiter, 
                                 column_label=self.column_label,
                                 column_text=self.column_text, 
                                 skip_first_row=self.skip_first_row, 
                                 labels=self.labels) 
        _train(
            task=self.task,
            logger=self.logger, tb_writer=self.tb_writer, model=self.model,
            tokenizer=self.tokenizer,dataset=dataset, max_steps=self.max_steps, 
            num_train_epochs=self.num_train_epochs, gradient_accumulation_steps=self.gradient_accumulation_steps, 
            weight_decay=self.weight_decay, learning_rate=self.learning_rate, 
            adam_epsilon=self.adam_epsilon, max_grad_norm=self.max_grad_norm,
            warmup_steps=self.warmup_steps,
            fp16=self.fp16, fp16_opt_level=self.fp16_opt_level, n_gpu=self.n_gpu,
            local_rank=self.local_rank, evaluate_during_training=self.evaluate_during_training,
            evaluate_func=self.evaluate,
            per_gpu_train_batch_size=self.per_gpu_train_batch_size, device=self.device,
            output_dir=self.output_dir, model_type=self.model_type, 
            model_name_or_path=self.model_name_or_path, configs=self.configs, seed=self.seed,
            logging_steps=self.logging_steps, save_steps=self.save_steps
        ) #

    def evaluate(self, 
                 mode,
                 tokenizer,
                 model): 
        """
           Evaluation Function, mostly call by train and predict.

           Args:
               mode: "train", "dev",  "test"
               tokenzier: tokenizer
               model: model
        """
        CACHE_PARAMS["mode"] = mode 
        eval_dataset = self.get_data(delimiter=self.delimiter, 
                                      column_label=self.column_label,
                                      column_text=self.column_text, 
                                      skip_first_row=self.skip_first_row, 
                                      labels=self.labels)
        return _evaluate(task=self.task, logger=self.logger, 
                   model_type=self.model_type, tokenizer=tokenizer, model=model,
                   dataset=eval_dataset, labels=self.labels, pad_token_label_id=self.pad_token_label_id,
                   per_gpu_eval_batch_size=self.per_gpu_eval_batch_size, n_gpu=self.n_gpu,
                   local_rank=self.local_rank, device=self.device, ouput_index=1)

    def predict(self):
        CACHE_PARAMS["mode"] = "test"
        _predict(task=self.task,
                    evaluate_func=self.evaluate, 
                    logger=self.logger, 
                    result_dir=self.result_dir,
                    data_dir=self.data_dir, 
                    output_dir=self.output_dir,
                    test_file_name="test.csv",
                    delimiter=self.delimiter, 
                    column_text=1, 
                    do_lower_case=self.do_lower_case, 
                    tokenizer_class=self.tokenizer_class, 
                    model_class=self.model_class, 
                    device=self.device,
                    prediction_model_dir=self.prediction_model_dir)
        

    def __call__(self):
        self.prepare_all()
        if self.do_train:
            self.train()
        if self.do_predict and self.local_rank in [-1, 0]:
            self.predict()



SUPPORTED_PIPELINE ={
      "ner": NerPipeline,
      "text-classification": TextClassificationPipeline,
  }
