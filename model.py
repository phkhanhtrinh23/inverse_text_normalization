# from transformers.file_utils import get_file_from_repo
from importlib.machinery import SourceFileLoader
import os
from transformers import EncoderDecoderModel, AutoConfig, AutoModel, AutoTokenizer, EncoderDecoderConfig, RobertaForCausalLM
from transformers.modeling_utils import PreTrainedModel, logging
import torch
from transformers.modeling_outputs import ModelOutput, CausalLMOutputWithCrossAttentions
from typing import Dict, Any, Optional, Tuple
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass
from model_config import InvertTextNormalizationConfig, PretrainedConfig, DecoderInvertTextNormalizationConfig

cache_dir = './cache'
encoder_model_name = 'vinai/phobert-base'
decoder_model_name = 'vinai/phobert-base'

if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
logger = logging.get_logger(__name__)


@dataclass
class InvertTextNormalizationOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    logits_spoken_tagging: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


def invert_text_features(encoder_hidden_states, word_src_lengths, spoken_label):
    list_features = []
    list_features_mask = []
    max_length = word_src_lengths.max()
    feature_pad = torch.zeros_like(encoder_hidden_states[0, :1, :])
    for hidden_state, word_length, list_idx in zip(encoder_hidden_states, word_src_lengths, spoken_label):
        for idx in list_idx:
            if idx > 0:
                start = sum(word_length[:idx])
                end = start + word_length[idx]
                remain_length = max_length - word_length[idx]
                list_features_mask.append(torch.cat([torch.ones_like(spoken_label[0, 0]).expand(word_length[idx]),
                                                     torch.zeros_like(
                                                         spoken_label[0, 0].expand(remain_length))]).unsqueeze(0))
                spoken_phrases_feature = hidden_state[start: end]

                list_features.append(torch.cat([spoken_phrases_feature,
                                                feature_pad.expand(remain_length, feature_pad.size(-1))]).unsqueeze(0))
    return torch.cat(list_features), torch.cat(list_features_mask)


def invert_text_labels(decoder_input_ids, labels, word_tgt_lengths, spoken_idx):
    list_decoder_input_ids = []
    list_labels = []
    max_length = word_tgt_lengths.max()
    init_decoder_ids = torch.tensor([0], device=labels.device, dtype=labels.dtype)
    pad_decoder_ids = torch.tensor([1], device=labels.device, dtype=labels.dtype)
    eos_decoder_ids = torch.tensor([2], device=labels.device, dtype=labels.dtype)
    ignore_labels = torch.tensor([-100], device=labels.device, dtype=labels.dtype)

    for decoder_inputs, decoder_label, word_length, list_idx in zip(decoder_input_ids,
                                                                    labels, word_tgt_lengths, spoken_idx):
        for idx in list_idx:
            if idx > 0:
                start = sum(word_length[:idx - 1])
                end = start + word_length[idx - 1]
                remain_length = max_length - word_length[idx - 1]
                remain_decoder_input_ids = max_length - len(decoder_inputs[start + 1:end + 1])
                list_decoder_input_ids.append(torch.cat([init_decoder_ids,
                                                         decoder_inputs[start + 1:end + 1],
                                                         pad_decoder_ids.expand(remain_decoder_input_ids)]).unsqueeze(0))
                list_labels.append(torch.cat([decoder_label[start:end],
                                              eos_decoder_ids,
                                              ignore_labels.expand(remain_length)]).unsqueeze(0))

    decoder_input_ids = torch.cat(list_decoder_input_ids)
    labels = torch.cat(list_labels)

    return decoder_input_ids, labels


class InvertTextNormalization(EncoderDecoderModel):
    config_class = InvertTextNormalizationConfig

    def __init__(
            self,
            config: Optional[PretrainedConfig] = None,
            encoder: Optional[PreTrainedModel] = None,
            decoder: Optional[PreTrainedModel] = None,
    ):
        if config is None and (encoder is None or decoder is None):
            raise ValueError("Either a configuration or an encoder and a decoder has to be provided.")
        if config is None:
            config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        if config.decoder.cross_attention_hidden_size is not None:
            if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
                raise ValueError(
                    "If `cross_attention_hidden_size` is specified in the decoder's configuration, "
                    "it has to be equal to the encoder's `hidden_size`. "
                    f"Got {config.decoder.cross_attention_hidden_size} for `config.decoder.cross_attention_hidden_size` "
                    f"and {config.encoder.hidden_size} for `config.encoder.hidden_size`."
                )

        # initialize with config
        super().__init__(config)

        if encoder is None:
            from transformers.models.auto.modeling_auto import AutoModel

            encoder = AutoModel.from_config(config.encoder)

        if decoder is None:

            decoder = DecoderInvertTextNormalization._from_config(config.decoder)

        self.encoder = encoder
        self.decoder = decoder

        if self.encoder.config.to_dict() != self.config.encoder.to_dict():
            logger.warning(
                f"Config of the encoder: {self.encoder.__class__} is overwritten by shared encoder config: {self.config.encoder}"
            )
        if self.decoder.config.to_dict() != self.config.decoder.to_dict():
            logger.warning(
                f"Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config: {self.config.decoder}"
            )

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.encoder.config = self.config.encoder
        self.decoder.config = self.config.decoder

        # encoder outputs might need to be projected to different dimension for decoder
        if (
                self.encoder.config.hidden_size != self.decoder.config.hidden_size
                and self.decoder.config.cross_attention_hidden_size is None
        ):
            self.enc_to_dec_proj = torch.nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size)

        if self.encoder.get_output_embeddings() is not None:
            raise ValueError(
                f"The encoder {self.encoder} should not have a LM Head. Please use a model without LM Head"
            )

        # spoken tagging
        self.dropout = torch.nn.Dropout(0.3)

        # 0: "O", 1: "B", 2: "I"
        self.spoken_tagging_classifier = torch.nn.Linear(config.encoder.hidden_size, 3)

        # tie encoder, decoder weights if config set accordingly
        self.tie_weights()

        # FFN
        # self.FFN = torch.nn.Linear(config.decoder.hidden_size, config.decoder.vocab_size)

    @classmethod
    def from_encoder_decoder_pretrained(
            cls,
            encoder_pretrained_model_name_or_path: str = None,
            decoder_pretrained_model_name_or_path: str = None,
            *model_args,
            **kwargs
    ) -> PreTrainedModel:
        kwargs_encoder = {
            argument[len("encoder_"):]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # remove encoder, decoder kwargs from kwargs
        for key in kwargs_encoder.keys():
            del kwargs["encoder_" + key]
        for key in kwargs_decoder.keys():
            del kwargs["decoder_" + key]

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            if encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `encoder_model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_encoder:
                encoder_config = AutoConfig.from_pretrained(encoder_pretrained_model_name_or_path)
                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                    logger.info(
                        f"Initializing {encoder_pretrained_model_name_or_path} as a encoder model "
                        "from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False

                kwargs_encoder["config"] = encoder_config

            encoder = AutoModel.from_pretrained(encoder_pretrained_model_name_or_path, *model_args,
                                                **kwargs_encoder)

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            if decoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_decoder:
                decoder_config = DecoderInvertTextNormalizationConfig.from_pretrained(decoder_pretrained_model_name_or_path)
                if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
                    logger.info(
                        f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. "
                        f"Cross attention layers are added to {decoder_pretrained_model_name_or_path} "
                        f"and randomly initialized if {decoder_pretrained_model_name_or_path}'s architecture allows for "
                        "cross attention layers."
                    )
                    decoder_config.is_decoder = True
                    decoder_config.add_cross_attention = True

                kwargs_decoder["config"] = decoder_config

            if kwargs_decoder["config"].is_decoder is False or kwargs_decoder["config"].add_cross_attention is False:
                logger.warning(
                    f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. "
                    f"In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, "
                    "make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` "
                    "passed to `.from_encoder_decoder_pretrained(...)` are set to `True` or do not pass a "
                    "`decoder_config` to `.from_encoder_decoder_pretrained(...)`"
                )

            decoder = DecoderInvertTextNormalization.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)

        # instantiate config with corresponding kwargs
        config = InvertTextNormalizationConfig.from_encoder_decoder_configs(encoder.config, decoder.config, **kwargs)

        return cls(encoder=encoder, decoder=decoder, config=config)
    
    """
    return_dict (bool, optional):   True - return a ModelOutput.
                                    False - return a plain tuple.
    """

    def prepare_inputs_for_generation(
            self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, past=past)
        # print("decoder_input_ids: ", decoder_inputs["input_ids"])
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
        }
        return input_dict

    def forward(
            self,
            input_ids=None,##
            attention_mask=None,# ##
            decoder_input_ids=None,#
            decoder_attention_mask=None,#
            encoder_outputs=None,#
            past_key_values=None,#
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,##
            use_cache=None,#
            spoken_label=None,##
            word_src_lengths=None,##
            word_tgt_lengths=None,##
            spoken_idx=None,##
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            inputs_length=None,
            outputs=None,
            outputs_length=None,
            src=None,
            tgt=None,
            **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }
        spoken_tagging_output = None
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
            spoken_tagging_output = self.spoken_tagging_classifier(self.dropout(encoder_outputs[0]))

        encoder_hidden_states = encoder_outputs[0]

        # print("spoken_tagging_output: ", torch.argmax(encoder_outputs.spoken_tagging_output, dim=-1))

        if spoken_idx is not None:
            encoder_hidden_states, attention_mask = invert_text_features(encoder_hidden_states,
                                                                                    word_src_lengths,
                                                                                    spoken_idx)
            decoder_input_ids, labels = invert_text_labels(decoder_input_ids, labels,
                                                                            word_tgt_lengths,
                                                                            spoken_idx)
        if (
                self.encoder.config.hidden_size != self.decoder.config.hidden_size
                and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        loss = None
        # if labels is not None:
        #     logits = self.FFN(self.dropout(decoder_outputs.last_hidden_state))
        #     loss_fct = CrossEntropyLoss()
        #     loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))

        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[1]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))

        if spoken_label is not None:
            loss_fct = CrossEntropyLoss()
            spoken_tagging_loss = loss_fct(spoken_tagging_output.reshape(-1, 3), spoken_label.view(-1))
            loss = loss + spoken_tagging_loss
        
        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return InvertTextNormalizationOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            logits_spoken_tagging=spoken_tagging_output,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            # encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
        )

class DecoderInvertTextNormalization(RobertaForCausalLM):
    config_class = DecoderInvertTextNormalizationConfig

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config):
        super().__init__(config)
        self.dense_query_copy = torch.nn.Linear(config.hidden_size, config.hidden_size)

    """
    torch.bmm(input, mat2, *, out=None) -> Tensor: 
    Performs a batch matrix-matrix product of matrices stored in input and mat2. input and mat2 must be 3-D tensors 
    each containing the same number of matrices. If input is a (b x n x m) tensor, mat2 is a (b x m x p) tensor, out will be 
    a (b x n xs p)tensor.
    """

    def forward_copy_attention(self, query, values, values_mask):
        """
        :param query: batch * output_steps * hidden_state
        :param values: batch * max_encoder_steps * hidden_state
        :param values_mask: batch * output_steps * max_encoder_steps
        :return: batch * output_steps * hidden_state
        """
        dot_attn_score = torch.bmm(query, values.transpose(2, 1))
        attn_mask = (1 - values_mask.clone().unsqueeze(1)).bool()
        dot_attn_score.masked_fill_(attn_mask, -float('inf'))
        dot_attn_score = torch.softmax(dot_attn_score, dim=-1)
        result_attention = torch.bmm(dot_attn_score, values)
        return result_attention

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        # attention with input encoded
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # output copy attention
        query_copy = torch.relu(self.dense_query_copy(sequence_output))
        sequence_atten_copy_output = self.forward_copy_attention(query_copy,
                                                                 encoder_hidden_states,
                                                                 encoder_attention_mask)

       
        prediction_scores = self.lm_head(sequence_output + sequence_atten_copy_output)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return output

        result = CausalLMOutputWithCrossAttentions(
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

        return result

def init_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(encoder_model_name, use_fast=False)
    # tokenizer = SourceFileLoader("envibert.tokenizer",
    #                              os.path.join(cache_dir,
    #                                           'envibert_tokenizer.py')).load_module().RobertaTokenizer(cache_dir)
    tokenizer.model_input_names = ["input_ids",
                                   "attention_mask",
                                   "labels"]
    return tokenizer


def init_model():
    tokenizer = init_tokenizer()

    # set encoder decoder tying to True
    roberta = InvertTextNormalization.from_encoder_decoder_pretrained(encoder_model_name,
                                                                    decoder_model_name,
                                                                    tie_encoder_decoder=False)

    # set special tokens
    roberta.config.decoder_start_token_id = tokenizer.bos_token_id
    roberta.config.eos_token_id = tokenizer.eos_token_id
    roberta.config.pad_token_id = tokenizer.pad_token_id

    # sensible parameters for beam search
    # set decoding params
    roberta.config.max_length = 200
    roberta.config.early_stopping = True
    roberta.config.no_repeat_ngram_size = 3
    roberta.config.length_penalty = 2.0
    roberta.config.num_beams = 1
    roberta.config.vocab_size = roberta.config.encoder.vocab_size

    return roberta, tokenizer
