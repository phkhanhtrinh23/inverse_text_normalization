import datasets
import model
from transformers import PreTrainedTokenizerBase
from typing import Optional, Union, Any
from transformers.file_utils import PaddingStrategy
import re
import time
import random
from dataclasses import dataclass
import validators

tokenizer = None

@dataclass
class DataCollatorInvertTextNormalization:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def encode_list_string(self, list_text):
        text_tokenized = self.tokenizer(list_text)
        return self.tokenizer.pad(
            text_tokenized,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )

    def __call__(self, features, return_tensors=None):
        batch_src, batch_tgt = [], []
        for item in features:
            src_spans, tgt_spans = create_invert_text_norm(item['src'], item['tgt'])
            batch_src.append(src_spans)
            batch_tgt.append(tgt_spans)

        # print("Batch: ", batch_src, batch_tgt)

        features = preprocess_function({"src": batch_src, "tgt": batch_tgt})
        # print("Preprocess batch: ", features)

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["outputs"] for feature in features] if "outputs" in features[0].keys() else None
        spoken_labels = [feature["spoken_label"] for feature in features] if "spoken_label" in features[0].keys() else None
        spoken_idx = [feature["src_spoken_idx"] for feature in features] if "src_spoken_idx" in features[0].keys() else None

        word_src_lengths = [feature["inputs_length"] for feature in features] if "inputs_length" in features[0].keys() else None
        word_tgt_lengths = [feature["outputs_length"] for feature in features] if "outputs_length" in features[0].keys() else None

        # print("Spoken labels: ", spoken_labels)
        
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            max_src_length = max(len(l) for l in spoken_labels)
            max_spoken_idx_length = max(len(l) for l in spoken_idx)
            max_word_src_length = max(len(l) for l in word_src_lengths)
            max_word_tgt_length = max(len(l) for l in word_tgt_lengths)

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["outputs"]))
                remainder_word_tgt_length = [0] * (max_word_tgt_length - len(feature["outputs_length"]))
                remainder_spoken = [self.label_pad_token_id] * (max_src_length - len(feature["spoken_label"]))
                remainder_spoken_idx = [self.label_pad_token_id] * (max_spoken_idx_length - len(feature["src_spoken_idx"]))
                remainder_word_src_length = [0] * (max_word_src_length - len(feature["inputs_length"]))

                feature["labels"] = (
                    feature["outputs"] + [
                        self.tokenizer.eos_token_id] + remainder if padding_side == "right" else remainder + feature[
                        "outputs"] + [self.tokenizer.eos_token_id]
                )

                feature["spoken_label"] = [self.label_pad_token_id] + feature["spoken_label"] + [self.label_pad_token_id]
                feature["spoken_label"] = feature["spoken_label"] + remainder_spoken if padding_side == "right" else remainder_spoken + feature["spoken_label"]
                feature["src_spoken_idx"] = feature["src_spoken_idx"] + remainder_spoken_idx

                feature['inputs_length'] = [1] + feature['inputs_length'] + [1]
                feature['outputs_length'] = feature['outputs_length'] + [1]

                feature["inputs_length"] = feature["inputs_length"] + remainder_word_src_length
                feature["outputs_length"] = feature["outputs_length"] + remainder_word_tgt_length

        # print("Features: ", features)

        features_inputs = [{
            "input_ids": [self.tokenizer.bos_token_id] + item["input_ids"] + [self.tokenizer.eos_token_id],
            "attention_mask": [self.tokenizer.pad_token_id] + item["attention_mask"] + [self.tokenizer.pad_token_id]
        } for item in features]
        features_inputs = self.tokenizer.pad(
            features_inputs,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        outputs = self.tokenizer.pad({"input_ids": [feature["labels"] for feature in features]},
                                     return_tensors=return_tensors)['input_ids']

        spoken_label = self.tokenizer.pad({"input_ids": [feature["spoken_label"] for feature in features]},
                                          return_tensors=return_tensors)['input_ids']

        spoken_idx = self.tokenizer.pad({"input_ids": [feature["src_spoken_idx"] for feature in features]},
                                        return_tensors=return_tensors)['input_ids'] + 1  # 1 for bos token

        word_src_lengths = self.tokenizer.pad({"input_ids": [feature["inputs_length"] for feature in features]},
                                              return_tensors=return_tensors)['input_ids']

        word_tgt_lengths = self.tokenizer.pad({"input_ids": [feature["outputs_length"] for feature in features]},
                                              return_tensors=return_tensors)['input_ids']

        features = {
            "input_ids": features_inputs["input_ids"],
            "spoken_label": spoken_label,
            "spoken_idx": spoken_idx,
            "word_src_lengths": word_src_lengths,
            "word_tgt_lengths": word_tgt_lengths,
            "attention_mask": features_inputs["attention_mask"],
            "labels": outputs,
        }

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features


def create_invert_text_norm(item_1, item_2):
    src_list, tgt_list = [], []
    for src, tgt in zip(item_1, item_2):
        if len(src) == len(tgt):
            if random.random() < 0.1:
                src_list.append(src)
                tgt_list.append(tgt)
        else:
            src_list.append(src)
            tgt_list.append(tgt)
    
    return src_list, tgt_list

# data init
def init_data():
    dataset = datasets.load_dataset('VietAI/spoken_norm_assignment')

    print("Dataset: ", dataset)
    return dataset


def preprocess_function(batch):

    global tokenizer
    if tokenizer is None:
        tokenizer = model.init_tokenizer()

    features = []
    for src_words, tgt_words in zip(batch["src"], batch["tgt"]):
        src_ids, pad_ids, src_lengths, tgt_ids, tgt_lengths = [], [], [], [], []
        # 0: "O", 1: "B", 2: "I"
        src_spoken_label = []  

        src_spoken_idx = []
        tgt_spoken_ids = []

        for idx, (src, tgt) in enumerate(zip(src_words, tgt_words)):
            # print("Src, Tgt: ", src, tgt)
            
            is_remain = False
            if src == tgt:
                is_remain = True

            src_tokenized = tokenizer(src)
            # print("Src tokenized: ", src_tokenized)
            
            if len(src_tokenized['input_ids']) < 3:
                continue

            # hardcode fix tokenizer email
            if validators.email(tgt):
                tgt_tokenized = tokenizer(tgt.replace('@', ' @'))
            else:
                tgt_tokenized = tokenizer(tgt)
            
            if len(tgt_tokenized['input_ids']) < 3:
                continue
            
            src_ids.extend(src_tokenized["input_ids"][1:-1])
            
            if is_remain:
                src_spoken_label.extend([0 if random.random() < 0.5 else -100 for _ in range(len(src_tokenized["input_ids"][1:-1]))])
                if random.random() < 0.1:
                    # Random pick normal word for spoken norm
                    src_spoken_idx.append(idx)
                    tgt_spoken_ids.append(tgt_tokenized["input_ids"][1:-1])
            else:
                src_spoken_label.extend([1] + [2] * (len(src_tokenized["input_ids"][1:-1]) - 1))
                src_spoken_idx.append(idx)
                tgt_spoken_ids.append(tgt_tokenized["input_ids"][1:-1])

            pad_ids.extend(src_tokenized["attention_mask"][1:-1])
            src_lengths.append(len(src_tokenized["input_ids"]) - 2)
            tgt_ids.extend(tgt_tokenized["input_ids"][1:-1])
            tgt_lengths.append(len(tgt_tokenized["input_ids"]) - 2)
            
            if len(src_ids) > 80 or len(tgt_ids) > 80:
                # print("Ignore sample")
                break

        if len(src_ids) < 1 or len(tgt_ids) < 1:
            continue

        features.append({
            "input_ids": src_ids,
            "attention_mask": pad_ids,
            "spoken_label": src_spoken_label,
            "inputs_length": src_lengths,
            "outputs": tgt_ids,
            "outputs_length": tgt_lengths,
            "src_spoken_idx": src_spoken_idx,
            "tgt_spoken_ids": tgt_spoken_ids
        })

    return features


if __name__ == "__main__":
    split_datasets = init_data()

    # model, model_tokenizer = model_handling.init_model()
    # data_collator = DataCollatorForNormSeq2Seq(model_tokenizer, model=model)
    if tokenizer is None:
        tokenizer = model.init_tokenizer()
    data_collator = DataCollatorInvertTextNormalization(tokenizer=tokenizer)
    import time
    start = time.time()
    # batch = data_collator([split_datasets["train"][i] for i in [random.randint(0, 900) for _ in range(0, 64)]])
    # print(batch)
    print("Sample 0: ", split_datasets["train"][0])
    data_collator([split_datasets["train"][0]])
    # print("{}s".format(time.time() - start))
    # print(split_datasets)