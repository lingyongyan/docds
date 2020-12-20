# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_bert import BertPreTrainedModel, BertEncoder, BertPooler, BertLayerNorm

from modeling_utils import sequence_mask, masked_logits, dsloss


class MyBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(MyBertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.entity_type_embeddings = nn.Embedding(2, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None,
                position_ids=None, entity_type_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long,
                                        device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if entity_type_ids is None:
            entity_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        entity_type_embeddings = self.entity_type_embeddings(entity_type_ids)

        embeddings = words_embeddings + position_embeddings + \
            token_type_embeddings + entity_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MyBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(MyBertModel, self).__init__(config)

        self.embeddings = MyBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings,
                                                      new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, entity_type_ids=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if entity_type_ids is None:
            entity_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids,
                                           token_type_ids=token_type_ids,
                                           entity_type_ids=entity_type_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

class BertForRelationExtraction(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForRelationExtraction, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = MyBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, candidate_index, candidate_length, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, entity_type_ids=None, answer_mask=None, loss_func=None,
                risk_sensitive=False, lambda_weight=0.5, gamma=1.0):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            entity_type_ids=entity_type_ids,
                            head_mask=head_mask)
        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        candidate_start_index, candidate_end_index = candidate_index.split(1, dim=-2)
        candidate_start_index = candidate_start_index.squeeze(-2)
        candidate_end_index = candidate_end_index.squeeze(-2)
        candidate_start_logits = start_logits.gather(-1, candidate_start_index)
        candidate_end_logits = end_logits.gather(-1, candidate_end_index)
        candidate_logits = candidate_start_logits + candidate_end_logits
        max_len = candidate_index.size(-1)
        candidate_mask = sequence_mask(candidate_length, max_len=max_len)

        masked_candidate_logits = masked_logits(candidate_logits, candidate_mask)

        outputs = (masked_candidate_logits,) + outputs[2:]

        if answer_mask is not None:
            if loss_func is None:
                loss_func = dsloss
            func_name = loss_func.__name__
            if func_name == 'dsloss':
                probs = F.softmax(masked_candidate_logits, dim=-1)
                total_loss = loss_func(probs,
                                       answer_mask.float(),
                                       candidate_mask.float(),
                                       risk_sensitive=risk_sensitive,
                                       lambda_weight=lambda_weight,
                                       gamma=gamma)
            else:
                log_probs = F.log_softmax(masked_candidate_logits, dim=-1)
                answer_mask = answer_mask.bool()
                total_loss = loss_func(-log_probs,
                                       answer_mask.float(),
                                       candidate_mask.float(),
                                       risk_sensitive=risk_sensitive,
                                       gamma=gamma)
            outputs = (total_loss, ) + outputs
        return outputs
