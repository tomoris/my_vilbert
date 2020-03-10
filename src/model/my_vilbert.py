# -*- coding: utf-8 -*-

# this program is extension of vilbert

from logging import getLogger

from typing import Tuple
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from .vilbert import (
    BertPreTrainingHeads,
    BertPreTrainedModel,
    BertEmbeddings,
    BertEncoder,
    BertTextPooler,
    BertImagePooler,
    BertLayerNorm,
)


logger = getLogger(__name__)


class MyBertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(MyBertModel, self).__init__(config)

        # initilize word embedding
        self.embeddings = BertEmbeddings(config)

        # initlize the vision embedding
        self.v_embeddings = MyBertNonTextEmbeddings(config)

        self.encoder = BertEncoder(config)
        self.t_pooler = BertTextPooler(config)
        self.v_pooler = BertImagePooler(config)

        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_txt,
        input_imgs,
        token_type_ids=None,
        attention_mask=None,
        image_attention_mask=None,
        co_attention_mask=None,
        output_all_encoded_layers=False,
        output_all_attention_masks=False,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_txt)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_txt)
        if image_attention_mask is None:
            image_attention_mask = torch.ones(
                input_imgs.size(0), input_imgs.size(1)
            ).type_as(input_txt)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_image_attention_mask = image_attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        extended_image_attention_mask = extended_image_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_image_attention_mask = (1.0 - extended_image_attention_mask) * -10000.0

        if co_attention_mask is None:
            co_attention_mask = torch.zeros(
                input_txt.size(0), input_imgs.size(1), input_txt.size(1)
            ).type_as(extended_image_attention_mask)

        extended_co_attention_mask = co_attention_mask.unsqueeze(1)

        # extended_co_attention_mask = co_attention_mask.unsqueeze(-1)
        extended_co_attention_mask = extended_co_attention_mask * 5.0
        extended_co_attention_mask = extended_co_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility

        embedding_output = self.embeddings(input_txt, token_type_ids)
        v_embedding_output = self.v_embeddings(input_imgs)

        encoded_layers_t, encoded_layers_v, all_attention_mask = self.encoder(
            embedding_output,
            v_embedding_output,
            extended_attention_mask,
            extended_image_attention_mask,
            extended_co_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            output_all_attention_masks=output_all_attention_masks,
        )

        sequence_output_t = encoded_layers_t[-1]
        sequence_output_v = encoded_layers_v[-1]

        pooled_output_t = self.t_pooler(sequence_output_t)
        pooled_output_v = self.v_pooler(sequence_output_v)

        if not output_all_encoded_layers:
            encoded_layers_t = encoded_layers_t[-1]
            encoded_layers_v = encoded_layers_v[-1]

        return (
            encoded_layers_t,
            encoded_layers_v,
            pooled_output_t,
            pooled_output_v,
            all_attention_mask,
        )


class MyBertNonTextEmbeddings(nn.Module):
    """Construct the embeddings from non-textual data.
    """

    def __init__(self, config):
        super(MyBertNonTextEmbeddings, self).__init__()

        self.non_text_embeddings = nn.Linear(
            config.v_feature_size, config.v_hidden_size
        )
        self.LayerNorm = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids):
        non_text_embeddings = self.non_text_embeddings(input_ids)
        embeddings = self.LayerNorm(non_text_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MyBertForNonTextPreTraining(BertPreTrainedModel):
    """BERT model with multi modal pre-training heads.
    """

    def __init__(self, config) -> None:
        super(MyBertForNonTextPreTraining, self).__init__(config)

        self.bert = MyBertModel(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight
        )
        self.next_sent_cls = nn.Linear(config.bi_hidden_size, 2)

        self.apply(self.init_bert_weights)
        self.predict_feature = config.predict_feature
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        # 元論文では90%マスクされた入力画像の意味クラスの分布と近づけるが、ここでは出力を入力に直接近づける。
        self.vis_criterion = nn.MSELoss(reduction="mean")
        if config.act_non_text == "relu":
            self.act_non_text = F.relu
        else:
            # non text feature に対する他の活性化関数を使いたいなら自分で足して
            # if you want to use other activation function for non textual feature,
            # you should add the activation function.
            raise NotImplementedError

    def forward(
        self,
        input_ids: Tensor,
        non_text_feat: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
        non_text_attention_mask: Tensor,
        masked_lm_labels: Tensor,
        next_sent_labels: Tensor,
        corresponding_text_and_non_text_label: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        (
            prediction_scores_t,
            prediction_scores_v,
            corresponding_score,
            next_sent_score,
        ) = self._forward(
            input_ids,
            non_text_feat,
            attention_mask,
            token_type_ids,
            non_text_attention_mask,
            masked_lm_labels,
            next_sent_labels,
            corresponding_text_and_non_text_label,
        )

        masked_lm_loss = self.loss_fct(
            prediction_scores_t.view(-1, self.config.vocab_size),
            masked_lm_labels.view(-1),
        )

        next_sent_loss = self.loss_fct(next_sent_score, next_sent_labels)

        prediction_scores_v = prediction_scores_v.squeeze(1)
        non_text_loss = self.vis_criterion(
            self.act_non_text(prediction_scores_v), non_text_feat
        )

        corresponding_loss = self.loss_fct(
            corresponding_score, corresponding_text_and_non_text_label,
        )

        return masked_lm_loss, next_sent_loss, non_text_loss, corresponding_loss

    def predict(
        self,
        input_ids: Tensor,
        non_text_feat: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
        non_text_attention_mask: Tensor,
        masked_lm_labels: Tensor,
        next_sent_labels: Tensor,
        corresponding_text_and_non_text_label: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        (
            prediction_scores_t,
            prediction_scores_v,
            corresponding_score,
            next_sent_score,
        ) = self._forward(
            input_ids,
            non_text_feat,
            attention_mask,
            token_type_ids,
            non_text_attention_mask,
            masked_lm_labels,
            next_sent_labels,
            corresponding_text_and_non_text_label,
        )

        argmax_prediction_scores_t = torch.argmax(prediction_scores_t, dim=2)
        argmax_next_sent_score = torch.argmax(next_sent_score, dim=1)
        argmax_corresponding_score = torch.argmax(corresponding_score, dim=1)
        return (
            argmax_prediction_scores_t,
            argmax_next_sent_score,
            argmax_corresponding_score,
        )

    def _forward(
        self,
        input_ids: Tensor,
        non_text_feat: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
        non_text_attention_mask: Tensor,
        masked_lm_labels: Tensor,
        next_sent_labels: Tensor,
        corresponding_text_and_non_text_label: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        # in this model, we first embed the images.
        (
            sequence_output_t,
            sequence_output_v,
            pooled_output_t,
            pooled_output_v,
            all_attention_mask,
        ) = self.bert(
            input_ids,
            (non_text_feat * non_text_attention_mask).unsqueeze(1),
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            image_attention_mask=None,
            output_all_encoded_layers=False,
            output_all_attention_masks=False,
        )

        prediction_scores_t, prediction_scores_v, corresponding_score = self.cls(
            sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v
        )

        next_sent_score = self.next_sent_cls(pooled_output_t)

        return (
            prediction_scores_t,
            prediction_scores_v,
            corresponding_score,
            next_sent_score,
        )

