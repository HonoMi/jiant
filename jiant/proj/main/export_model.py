import os
from typing import Tuple, Type
import logging

import torch
import transformers

import jiant.utils.python.io as py_io
import jiant.utils.zconf as zconf

logger = logging.getLogger(__name__)


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    model_type = zconf.attr(type=str)
    output_base_path = zconf.attr(type=str)
    hf_model_name = zconf.attr(type=str, default=None)


def lookup_and_export_model(model_type: str, output_base_path: str, hf_model_name: str = None):
    model_class, tokenizer_class = get_model_and_tokenizer_classes(model_type)
    export_model(
        model_type=model_type,
        output_base_path=output_base_path,
        model_class=model_class,
        tokenizer_class=tokenizer_class,
        hf_model_name=hf_model_name,
    )


def export_model(
    model_type: str,
    output_base_path: str,
    model_class: Type[transformers.PreTrainedModel],
    tokenizer_class: Type[transformers.PreTrainedTokenizer],
    hf_model_name: str = None,
    skip_if_exists: bool = True,
):
    """Retrieve model and tokenizer from Transformers and save all necessary data
    Things saved:
    - Model weights
    - Model config JSON (corresponding to corresponding Transformers model Config object)
    - Tokenizer data
    - JSON file pointing to paths for the above
    Args:
        model_type: Model-type string. See: `get_model_and_tokenizer_classes`
        output_base_path: Base path to save output to
        model_class: Model class
        tokenizer_class: Tokenizer class
        hf_model_name: (Optional) hf_model_name from https://huggingface.co/models,
                       if it differs from model_type
    """
    if hf_model_name is None:
        hf_model_name = model_type

    model_fol_path = os.path.join(output_base_path, "model")

    model_path = os.path.join(model_fol_path, f"{model_type}.p")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Necessary since some models are named "facebook/bart-base"
    model = model_class.from_pretrained(hf_model_name)
    with py_io.get_lock(model_path):
        if skip_if_exists and os.path.exists(model_path):
            logger.info('Skip writing to %s since it already exists.', model_path)
        else:
            torch.save(model.state_dict(), model_path)

    model_config_path = os.path.join(model_fol_path, f"{model_type}.json")
    os.makedirs(os.path.dirname(model_config_path), exist_ok=True)
    py_io.write_json(model.config.to_dict(), model_config_path, skip_if_exists=skip_if_exists)

    tokenizer_fol_path = os.path.join(output_base_path, "tokenizer")
    # os.makedirs(tokenizer_fol_path, exist_ok=True)
    tokenizer = tokenizer_class.from_pretrained(hf_model_name)
    with py_io.get_lock(tokenizer_fol_path):
        if skip_if_exists and os.path.exists(tokenizer_fol_path):
            logger.info('Skip writing to %s since it already exists.', tokenizer_fol_path)
        else:
            tokenizer.save_pretrained(tokenizer_fol_path)

    config = {
        "model_type": model_type,
        "model_path": model_path,
        "model_config_path": model_config_path,
        "model_tokenizer_path": tokenizer_fol_path,
    }
    py_io.write_json(config, os.path.join(output_base_path, f"config.json"), skip_if_exists=skip_if_exists)


def get_model_and_tokenizer_classes(
    model_type: str,
) -> Tuple[Type[transformers.PreTrainedModel], Type[transformers.PreTrainedTokenizer]]:
    # We want the chosen model to have all the weights from pretraining (if possible)
    class_lookup = {
        # HONOKA: 学習済みweightの上に，何らかの層があればよい．
        # 生成モデル以外は，ForPreTraining -> ForMaskedLM -> LMHeadModel の順に，存在するクラスを使っているようにみえる．
        "bert": (transformers.BertForPreTraining, transformers.BertTokenizer),
        "xlm-clm-": (transformers.XLMWithLMHeadModel, transformers.XLMTokenizer),
        "roberta": (transformers.RobertaForMaskedLM, transformers.RobertaTokenizer),
        "albert": (transformers.AlbertForMaskedLM, transformers.AlbertTokenizer),
        "facebook/bart": (transformers.BartForConditionalGeneration, transformers.BartTokenizer),
        "facebook/mbart": (transformers.BartForConditionalGeneration, transformers.MBartTokenizer),
        "google/electra": (transformers.ElectraForPreTraining, transformers.ElectraTokenizer),

        # 以下，まだ使えない．他のファイルを編集していないから．
        "gpt2": (transformers.GPT2LMHeadModel, transformers.GPT2Tokenizer),
        "transformer": (transformers.TransfoXLLMHeadModel, transformers.TransfoXLTokenizer),
        "xlnet": (transformers.XLNetLMHeadModel, transformers.XLNetTokenizer),
    }
    if model_type.split("-")[0] in class_lookup:
        return class_lookup[model_type.split("-")[0]]
    elif model_type.startswith("xlm-mlm-") or model_type.startswith("xlm-clm-"):
        return transformers.XLMWithLMHeadModel, transformers.XLMTokenizer
    elif model_type.startswith("xlm-roberta-"):
        return transformers.XLMRobertaForMaskedLM, transformers.XLMRobertaTokenizer
    else:
        raise KeyError()


def main():
    args = RunConfiguration.default_run_cli()
    lookup_and_export_model(
        model_type=args.model_type,
        output_base_path=args.output_base_path,
        hf_model_name=args.hf_model_name,
    )


if __name__ == "__main__":
    main()
