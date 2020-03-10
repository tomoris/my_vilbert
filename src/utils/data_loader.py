from logging import getLogger
import random

from typing import List, Tuple, Dict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

logger = getLogger(__name__)


def pad_and_corresponding_non_text_collate(
    batch: List[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    token_ids_list = [item[0] for item in batch]
    masked_lm_labels_list = [item[1] for item in batch]
    token_type_ids_list = [item[2] for item in batch]
    next_sent_labels_list = [item[3] for item in batch]
    non_text_feat_list = [item[4].expand(1, -1) for item in batch]
    non_text_attention_list = [item[5].expand(1, -1) for item in batch]
    coresponding_label_list = [item[6] for item in batch]

    pad_list: List[Tensor] = [
        torch.ones(_.size(0), dtype=torch.uint8) for _ in token_ids_list
    ]

    token_ids = pad_sequence(token_ids_list, batch_first=True)
    masked_lm_labels = pad_sequence(
        masked_lm_labels_list, batch_first=True, padding_value=-1
    )
    pad_ids = pad_sequence(pad_list, batch_first=True, padding_value=0)
    token_type_ids = pad_sequence(
        token_type_ids_list, batch_first=True, padding_value=1
    )
    next_sent_labels = torch.cat(next_sent_labels_list, dim=0)
    non_text_feat = torch.cat(non_text_feat_list, dim=0)
    non_text_attention = torch.cat(non_text_attention_list, dim=0)
    coresponding_labels = torch.cat(coresponding_label_list, dim=0)

    assert (
        token_ids.size()
        == masked_lm_labels.size()
        == pad_ids.size()
        == token_type_ids.size()
    )
    assert non_text_feat.size() == non_text_attention.size()
    assert token_ids.size(0) == non_text_feat.size(0)
    assert token_ids.size(0) == coresponding_labels.size(0)
    assert next_sent_labels.size() == coresponding_labels.size()

    new_batch = (
        token_ids,
        pad_ids,
        masked_lm_labels,
        token_type_ids,
        next_sent_labels,
        non_text_feat,
        non_text_attention,
        coresponding_labels,
    )
    return new_batch


class MyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file_name: str,
        tokenizer,
        max_len: int,
        non_text_feature_file_name: str,
        train_next_sentence: bool,
    ) -> None:
        self.max_token_len = max_len
        self.corpus: List[Tensor] = []
        self.doc_idx2sent_start_idx: Dict[int, int] = {}
        self.sent_idx2doc_idx: Dict[int, int] = {}
        self.sent2non_text: List[int] = []
        self.non_text_data: Tensor
        self.tokenizer = tokenizer
        self.max_len: int = max_len
        self.train_next_sentence: bool = train_next_sentence

        self.corpus_load(file_name)
        self.non_text_load(non_text_feature_file_name)

        # self.corpusの0番目からtrain_end番目までを学習データとする
        doc_idx = self.sent_idx2doc_idx[int(len(self.corpus) * 0.99) - 1] - 1
        self.train_end: int = self.doc_idx2sent_start_idx[doc_idx] - 1
        self.train = True

    def __len__(self) -> int:
        if self.train:
            return self.train_end
        else:
            return len(self.corpus[self.train_end + 1 :])

    def mask_text(self, token_ids: Tensor) -> Tuple[Tensor, Tensor]:
        sent_len = token_ids.size(0)
        rand_choice = random.sample([i for i in range(sent_len)], int(sent_len * 0.15))
        masked_lm_labels = torch.ones(sent_len, dtype=torch.long) * -1
        for mask_token_pos in rand_choice:
            masked_lm_labels[mask_token_pos] = token_ids[mask_token_pos]
            r = random.uniform(0.0, 1.0)
            if r < 0.8:
                token_ids[mask_token_pos] = self.tokenizer.vocab["[MASK]"]
            elif r < 0.9:
                rand_token = random.randint(0, self.tokenizer.vocab_size - 1)
                token_ids[mask_token_pos] = rand_token
            else:
                pass
        return token_ids, masked_lm_labels

    def __getitem__(
        self, i: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if self.train:
            pass
        else:
            i = i + self.train_end + 1
        # 入力テキストの一部をマスク化
        token_ids = self.corpus[i].clone()
        token_ids, masked_lm_labels = self.mask_text(token_ids)
        token_ids = token_ids[: self.max_len]
        masked_lm_labels = masked_lm_labels[: self.max_len]
        if self.train_next_sentence:
            # 正しい次の文か間違った次の文をサンプリング
            next_sent_label: Tensor
            r = random.uniform(0.0, 1.0)
            if r > 0.5:
                # 正しい次文
                doc_idx = self.sent_idx2doc_idx[i]
                if doc_idx < len(self.doc_idx2sent_start_idx) - 1:
                    doc_end = self.doc_idx2sent_start_idx[doc_idx + 1] - 1
                else:
                    doc_end = len(self.corpus) - 1
                if i + 1 <= doc_end:
                    next_sent_idx = i + 1
                    next_sent_label = torch.tensor([1], dtype=torch.long)
                else:
                    next_sent_idx = random.randint(0, len(self) - 1)
                    next_sent_label = torch.tensor([0], dtype=torch.long)
            else:
                # 間違った文
                next_sent_idx = random.randint(0, len(self) - 1)
                next_sent_label = torch.tensor([0], dtype=torch.long)
            next_token_ids = self.corpus[next_sent_idx].clone()
            next_token_ids, next_masked_lm_labels = self.mask_text(next_token_ids)
            token_type_ids = torch.cat(
                [
                    torch.zeros(token_ids.size(0), dtype=torch.long),
                    torch.ones(next_token_ids.size(0), dtype=torch.long),
                ],
                dim=0,
            )[: self.max_len]
            token_ids = torch.cat([token_ids, next_token_ids], dim=0)[: self.max_len]
            masked_lm_labels = torch.cat(
                [masked_lm_labels, next_masked_lm_labels], dim=0
            )[: self.max_len]
        else:
            token_type_ids = torch.zeros(token_ids.size(0), dtype=torch.long)

        # non text の入力を決める
        # まずはテキストと対応するかどうかを
        r = random.uniform(0.0, 1.0)
        non_text_feat: Tensor
        non_text_attention: Tensor
        coresponding_label: Tensor
        if r > 0.5:
            # 対応する
            coresponding_label = torch.tensor([1], dtype=torch.long)
            non_text_idx = self.sent2non_text[i]
            if non_text_idx != -1:
                non_text_feat = self.non_text_data[non_text_idx].clone()
            else:
                non_text_feat = torch.zeros(
                    self.non_text_data.size(1), dtype=self.non_text_data.dtype
                )
        else:
            # 対応しない
            coresponding_label = torch.tensor([0], dtype=torch.long)
            non_text_data_size = self.non_text_data.size(0)
            r = random.randint(0, non_text_data_size - 1)
            non_text_feat = self.non_text_data[r].clone()
        # 15%の確率でnon textual data にマスクをかける。
        r = random.uniform(0.0, 1.0)
        if r < 0.15:
            # 90%程度を0に。torch.randは[0, 1)の間からunifromな値
            ones = torch.ones(self.non_text_data.size(1))
            zeros = torch.zeros(self.non_text_data.size(1))
            rands = torch.rand(self.non_text_data.size(1))
            non_text_attention = torch.where(rands > ones * 0.9, zeros, ones)
        else:
            non_text_attention = torch.ones(self.non_text_data.size(1))

        item = (
            token_ids,
            masked_lm_labels,
            token_type_ids,
            next_sent_label,
            non_text_feat,
            non_text_attention,
            coresponding_label,
        )
        return item

    def corpus_load(self, file_name: str) -> None:
        logger.debug("load text data")
        num_lines = sum(1 for line in open(file_name, "r"))
        self.doc_idx2sent_start_idx[0] = 0
        with open(file_name, "r") as f:
            for line in tqdm(f, total=num_lines):
                line = line.rstrip()
                if line == "":
                    self.doc_idx2sent_start_idx[len(self.doc_idx2sent_start_idx)] = len(
                        self.corpus
                    )
                    continue
                line_sp = line.split("\t")
                assert len(line_sp) == 2
                tokenized_sent = self.tokenizer.tokenize(line_sp[0])
                indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_sent)
                torch_indexed_tokens = torch.tensor(
                    indexed_tokens[: self.max_token_len]
                )
                self.corpus.append(torch_indexed_tokens)
                self.sent2non_text.append(int(line_sp[1]))
                self.sent_idx2doc_idx[len(self.corpus) - 1] = len(
                    self.doc_idx2sent_start_idx
                )

    def non_text_load(self, non_text_feature_file_name: str) -> None:
        logger.debug("load non-textual data")
        self.non_text_data = np.load(non_text_feature_file_name)
        self.non_text_data = torch.tensor(self.non_text_data)


def get_data_loader(
    file_name: str,
    tokenizer,
    max_len: int,
    non_text_feature_file_name: str,
    batch_size: int,
    train_next_sentence: bool,
) -> Tuple[DataLoader, MyDataset]:
    dataset = MyDataset(
        file_name, tokenizer, max_len, non_text_feature_file_name, train_next_sentence
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_and_corresponding_non_text_collate,
    )
    return data_loader, dataset
