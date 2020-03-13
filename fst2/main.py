import yaml
import os
from abc import ABC, abstractclassmethod
from argparse import ArgumentParser
from .utils import gen_configs, gen_dirs
from .pipelines import pipeline


LOGO = """
           ++++++       ***     +++++++
           ++         *            +
           ++++++       ***        +
           +               *       +
           +           ****        +
"""


print(LOGO)

###################### SubCommands  ###################
class BaseSubcommand(ABC):
    @abstractclassmethod
    def rigister_subcommnd(self):
        raise NotImplementedError

### Prepare Subcommand Section ###
class PrepareCommand(BaseSubcommand):
    @staticmethod
    def rigister_subcommnd(subparser):
        prepare_parser = subparser.add_parser("prepare", help="CLI tool to help to gennerate a recommended yaml config file and directory for NLP trainning based on specific tasks")
        prepare_parser.add_argument("--gen-config", action="store_true", 
                                                   help="generate a yaml for train interactively")
        prepare_parser.add_argument("--gen-dir", 
                                        action="store_true", 
                                        help="generate a recommended \
                                            dirctory for train")
        prepare_parser.add_argument("--parent-dirname", default="fst", help="pararent dirctory name")                                          
        prepare_parser.set_defaults(func=prepare)

def prepare(args):
    if args.gen_config:
        gen_configs(args)
    if args.gen_dir:
        gen_dirs(args)


### Train Subcommand Section  ###
def train(args):
    config_file = os.path.join(args.config_dir, "configs.yml")
    if not os.path.exists(config_file):
        exit("make sure  configs.yml under current directoy ")
    # use pipeline to chose a train Pipeline for a given nlp task
    with open("configs.yml", "r", encoding="utf-8") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    # make sure dtype of params are all right, 科学计数法变str
    # TODO: add check configs functionality
    nlp = pipeline(configs)
    nlp() 


class StartCommand(BaseSubcommand):
    @staticmethod
    def rigister_subcommnd(subparser):
        """
        """
        train_parser = subparser.add_parser("start", help="CLI tool to train NLP models based on a yaml comifgs")
        #train_parser.add_argument("--do-train", action="store_true", help="train the model") #
        train_parser.add_argument("--config-dir", default=".", help="directory contains the configs.yml file")
        train_parser.set_defaults(func=train)



### Serve SubCommand Section ###
class ServeCommand(BaseSubcommand):
    pass


def entry():
    parser = ArgumentParser()  # directory name是一个问题
    
    subcommand = parser.add_subparsers()
    PrepareCommand.rigister_subcommnd(subcommand)
    StartCommand.rigister_subcommnd(subcommand)
    
    args = parser.parse_args()
    if not hasattr(args, 'func'):
        parser.print_help()
        exit(1)
    args.func(args)
    

if __name__ == "__main__":
    entry()

