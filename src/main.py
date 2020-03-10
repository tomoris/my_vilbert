from logging import getLogger
import argparse
import json
import os
import random

from typing import List, Any, Tuple
import torch
from torch import Tensor

from transformers import BertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from model.vilbert import BertConfig

from utils.logger_config import load_logger_config
from utils.data_loader import get_data_loader
from model.my_vilbert import MyBertForNonTextPreTraining

seed = 12345
random.seed(seed)
torch.manual_seed(seed)

logger = getLogger(__name__)


def eval(
    argmax_prediction_scores_t: Tensor,
    argmax_next_sent_score: Tensor,
    argmax_corresponding_score: Tensor,
    masked_lm_labels: Tensor,
    next_sent_labels: Tensor,
    corresponding_labels: Tensor,
):  # -> Tuple[int, int, int, int]:
    tp_mask_token = torch.eq(argmax_prediction_scores_t, masked_lm_labels).sum().item()
    tp_next_sent = torch.eq(argmax_next_sent_score, next_sent_labels).sum().item()
    tp_corresponding = (
        torch.eq(argmax_corresponding_score, corresponding_labels).sum().item()
    )
    non_mask = torch.ones_like(masked_lm_labels) * -1
    batch_size, sent_len = masked_lm_labels.size()
    count_masked_token = (batch_size * sent_len) - torch.eq(
        non_mask, masked_lm_labels
    ).sum().item()
    return tp_mask_token, tp_next_sent, tp_corresponding, count_masked_token


def pretrain(args) -> None:
    logger.info("start setup")

    # model setup
    config = BertConfig.from_json_file(args.config)
    model = MyBertForNonTextPreTraining.from_pretrained(args.from_pretrained, config)
    # model = MyBertForNonTextPreTraining(config)
    device = torch.device("cpu")
    if args.gpu >= 0:
        device = torch.device("cuda:" + str(args.gpu))
        model.to(device)
    logger.debug("device = {}".format(device))

    # tokenizer setup
    tokenizer = BertTokenizer.from_pretrained(
        args.vocab_file,
        do_lower_case=args.do_lower_case,
        tokenize_chinese_chars=args.tokenize_chinese_chars,
        max_len=args.max_len,
    )

    # data_loader setup
    train_next_sentence = json.load(open(args.config, "r"))["train_next_sentence"]
    logger.debug("train_next_sentence is {}".format(train_next_sentence))
    data_loader, data_set = get_data_loader(
        args.pretrain_text_file,
        tokenizer,
        args.max_len,
        args.pretrain_non_text_feature_file,
        args.train_batch_size,
        train_next_sentence,
    )
    train_data_size = len(data_set)

    # optimizer setup
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    bert_weight_name = json.load(open(args.bert_weight_name_file, "r"))
    optimizer_grouped_parameters: List[Any] = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if key[12:] in bert_weight_name:
                lr = args.learning_rate * 0.1
            else:
                lr = args.learning_rate
            if any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": lr, "weight_decay": 0.01}
                ]
            if not any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": lr, "weight_decay": 0.0}
                ]

    num_training_steps = len(data_loader) * args.num_training_epochs
    num_warmup_steps = int(args.warmup_proportion * num_training_steps)
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    logger.debug("train_data_size:{}".format(train_data_size))
    logger.debug("num_training_steps = {}".format(num_training_steps))
    logger.debug("num_warmup_steps = {}".format(num_warmup_steps))
    data_set.train = False
    logger.debug("dev_data_size:{}".format(len(data_set)))
    data_set.train = True

    # learning step
    logger.info("start learning")
    for epoch in range(args.num_training_epochs):
        model.train()
        data_set.trian = True
        total_loss = 0
        total_masked_lm_loss = 0
        total_next_sent_loss = 0
        total_non_text_loss = 0
        total_corresponding_loss = 0
        for i, batch in enumerate(data_loader):
            (
                token_ids,
                pad_ids,
                masked_lm_labels,
                token_type_ids,
                next_sent_labels,
                non_text_feat,
                non_text_attention,
                corresponding_labels,
            ) = batch
            if args.gpu >= 0:
                token_ids = token_ids.to(device)
                pad_ids = pad_ids.to(device)
                masked_lm_labels = masked_lm_labels.to(device)
                token_type_ids = token_type_ids.to(device)
                next_sent_labels = next_sent_labels.to(device)
                non_text_feat = non_text_feat.to(device)
                non_text_attention = non_text_attention.to(device)
                corresponding_labels = corresponding_labels.to(device)

            masked_lm_loss, next_sent_loss, non_text_loss, corresponding_loss = model(
                token_ids,
                non_text_feat,
                pad_ids,
                token_type_ids,
                non_text_attention,
                masked_lm_labels,
                next_sent_labels,
                corresponding_labels,
            )
            if train_next_sentence:
                loss = (
                    masked_lm_loss + next_sent_loss + non_text_loss + corresponding_loss
                )
            else:
                loss = masked_lm_loss + non_text_loss + corresponding_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            total_masked_lm_loss += masked_lm_loss.item()
            total_next_sent_loss += next_sent_loss.item()
            total_non_text_loss += non_text_loss.item()
            total_corresponding_loss += corresponding_loss.item()
            logger.debug("epoch:{0} i:{1} loss:{2}".format(epoch, i, loss.item()))
            logger.debug(
                "epoch:{0} i:{1} masked_lm_loss:{2}".format(
                    epoch, i, masked_lm_loss.item()
                )
            )
            logger.debug(
                "epoch:{0} i:{1} next_sent_loss:{2}".format(
                    epoch, i, next_sent_loss.item()
                )
            )
            logger.debug(
                "epoch:{0} i:{1} non_text_loss:{2}".format(
                    epoch, i, non_text_loss.item()
                )
            )
            logger.debug(
                "epoch:{0} i:{1} corresponding_loss:{2}".format(
                    epoch, i, corresponding_loss.item()
                )
            )
        logger.debug("epoch:{0} total_loss:{1}".format(epoch, total_loss))
        logger.debug(
            "epoch:{0} total_masked_lm_loss:{1}".format(epoch, total_masked_lm_loss)
        )
        logger.debug(
            "epoch:{0} total_next_sent_loss:{1}".format(epoch, total_next_sent_loss)
        )
        logger.debug(
            "epoch:{0} total_non_text_loss:{1}".format(epoch, total_non_text_loss)
        )
        logger.debug(
            "epoch:{0} total_corresponding_loss:{1}".format(
                epoch, total_corresponding_loss
            )
        )

        # development set で評価
        model.eval()
        data_set.trian = False
        total_tp_mask_token = 0
        total_tp_next_sent = 0
        total_tp_corresponding = 0
        total_count_masked_token = 0
        for i, batch in enumerate(data_loader):
            (
                token_ids,
                pad_ids,
                masked_lm_labels,
                token_type_ids,
                next_sent_labels,
                non_text_feat,
                non_text_attention,
                corresponding_labels,
            ) = batch
            if args.gpu >= 0:
                token_ids = token_ids.to(device)
                pad_ids = pad_ids.to(device)
                masked_lm_labels = masked_lm_labels.to(device)
                token_type_ids = token_type_ids.to(device)
                next_sent_labels = next_sent_labels.to(device)
                non_text_feat = non_text_feat.to(device)
                non_text_attention = non_text_attention.to(device)
                corresponding_labels = corresponding_labels.to(device)

            (
                argmax_prediction_scores_t,
                argmax_next_sent_score,
                argmax_corresponding_score,
            ) = model.predict(
                token_ids,
                non_text_feat,
                pad_ids,
                token_type_ids,
                non_text_attention,
                masked_lm_labels,
                next_sent_labels,
                corresponding_labels,
            )

            tp_mask_token, tp_next_sent, tp_corresponding, count_masked_token = eval(
                argmax_prediction_scores_t,
                argmax_next_sent_score,
                argmax_corresponding_score,
                masked_lm_labels,
                next_sent_labels,
                corresponding_labels,
            )
            total_tp_mask_token += tp_mask_token
            total_tp_next_sent += tp_next_sent
            total_tp_corresponding += tp_corresponding
            total_count_masked_token += count_masked_token
        accu_mask_token = float(total_tp_mask_token) / float(total_count_masked_token)
        accu_next_sent = float(total_tp_next_sent) / float(len(data_set))
        accu_corresponding = float(total_tp_corresponding) / float(len(data_set))
        logger.info("epoch:{0} accu_mask_token:{1:.4f}".format(epoch, accu_mask_token))
        logger.info("epoch:{0} accu_next_sent:{1:.4f}".format(epoch, accu_next_sent))
        logger.info(
            "epoch:{0} accu_corresponding:{1:.4f}".format(epoch, accu_corresponding)
        )

        # save
        save_file_name = args.save_path + "_epoch_" + str(epoch) + ".weight"
        torch.save(model.state_dict(), save_file_name)


def filetune(args):
    # model setup
    config = BertConfig.from_json_file(args.config)
    model = MyBertForNonTextPreTraining.from_pretrained(args.from_pretrained, config)
    if args.load_model_weight_file is None:
        assert False
    model.load_state_dict(torch.load(args.load_model_weight_file))
    device = torch.device("cpu")
    if args.gpu >= 0:
        device = torch.device("cuda:" + str(args.gpu))
        model.to(device)
    logger.debug("device = {}".format(device))

    # tokenizer setup
    tokenizer = BertTokenizer.from_pretrained(
        args.vocab_file,
        do_lower_case=args.do_lower_case,
        tokenize_chinese_chars=args.tokenize_chinese_chars,
        max_len=args.max_len,
    )

    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="my_vilbert")
    parser.add_argument(
        "--mode",
        help="pretrain or finetune",
        type=str,
        choices=["pretrain", "finetune"],
        required=True,
    )
    parser.add_argument(
        "--config", help="configuration file path", type=str, required=True
    )
    parser.add_argument(
        "--gpu", help="if you use cpu, please set -1", type=int, default=-1
    )
    parser.add_argument(
        "--from_pretrained",
        default="",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument("--bert_weight_name_file", type=str, required=True)
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="")
    parser.add_argument("--num_training_epochs", default=15, type=int, help="epoch")
    parser.add_argument(
        "--train_batch_size",
        default=16,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument("--vocab_file", default="", type=str, help="Bert vocab file")
    parser.add_argument(
        "--do_lower_case",
        type=bool,
        default=False,
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
    parser.add_argument("--tokenize_chinese_chars", type=bool, default=True, help="")
    parser.add_argument(
        "--max_len", type=int, default=128, help="maximum length of input tokens"
    )
    parser.add_argument(
        "--pretrain_text_file", type=str, help="pretraining file path", required=True
    )
    parser.add_argument(
        "--pretrain_non_text_feature_file",
        type=str,
        help="pretraining non-textual data file path. (npy format)",
        required=True,
    )
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--load_model_weight_file", type=str, default=None)
    args = parser.parse_args()

    load_logger_config(log_file_name=args.log_file)
    logger = getLogger(__name__)
    logger.debug("Start logger")
    logger.debug("server name = {}".format(os.uname().nodename))

    # log args parameters
    logger.debug("args parameters")
    for k, v in args._get_kwargs():
        logger.debug("{0} = {1}".format(k, v))
    # log config parameters
    logger.debug("config parameters")
    config = json.load(open(args.config, "r"))
    for k, v in config.items():
        logger.debug("{0} = {1}".format(k, v))

    if args.mode == "pretrain":
        pretrain(args)
    elif args.mode == "finetune":
        filetune(args)


if __name__ == "__main__":
    main()
