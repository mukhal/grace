from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass, field
from transformers import TrainingArguments

TASKS = ["gsm8k", "mathqa", "multiarith", "svamp", "last_letter_concatenation"]

@dataclass
class VerifierModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )

    pooling: Optional[str] = field(
        default="max",
        metadata={
            "help": (
                "The pooling strategy for the encoder outputs. Can be 'mean' or 'max'."
            )
        },
    )


@dataclass
class VerifierDataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    model_style : Optional[str] = field(
        default="enc",
        metadata={       
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        
        },
    )
    task: Optional[str] = field(
        default=None, metadata={"help": "The name of the task to train on: " + ", ".join(TASKS),
                                "choices": TASKS 
                            }
    )
    trajectory_path: Optional[str] = field(default=None, metadata={"help": "Path to the model trajectories file"})
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_len: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    n_examples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training and eval examples to this "
                "value if set."
            )
        },
    )
    dev_is_train: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "If true, the dev set is used as the train set. Useful for debugging purposes."
            )
        },
    )

    balance: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "If true, the training set is balanced."
            )
        },
    )

    invalid_prefix_prob: Optional[float] = field(
        default=0.0,
        metadata={
            "help": (
                "The probability of replacing a valid prefix with an invalid one."
            )
        },
    )

    max_alignment_cost: Optional[float] = field(
        default=1.8,
        metadata={
            "help": (
                "The probability of replacing a valid prefix with an invalid one."
            )
        },
    )



@dataclass
class VerifierTrainingArguments(TrainingArguments):
    ckpt_dir: Optional[str] = field(
        default=None,
        metadata={        
            "help": (
                "The checkpoint to load"
            )
        },
    )




    
