import torch
import model
from model import InvertTextNormalization
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tokenizer = model.init_tokenizer()
roberta = InvertTextNormalization.from_pretrained('checkpoints/checkpoint-18740', cache_dir=model.cache_dir)

list_inputs = ['ngày chín tháng một năm một chín bốn lăm nạn đói bùng phát ở Việt Nam và ảnh hưởng tới sáu mươi phần trăm dân',
            'tôi làm việc ở Khoa Khoa học và Kỹ thuật Máy tính và hôm nay tôi đã chấm được hai ngàn không trăm hai mươi ba bài',
            'tám giờ chín phút ngày mười tám tháng năm năm hai nghìn không trăm hai mươi hai',
            'mã số quy đê tê tê đê hai tám chéo hai không không ba',
            'thể tích tám mét khối trọng lượng năm mươi ki lô gam',
            'ngày hai tám tháng tư cô vít bùng phát ở Việt Nam gây nhiễm bệnh mười phần trăm dân Việt Nam'
            ]

"""
input  : ngày chín tháng một năm một chín bốn lăm nạn đói bùng phát ở Việt Nam và ảnh hưởng tới sáu mươi phần trăm dân
output : ngày 9/1/1945 nạn đói bùng phát ở Việt Nam và ảnh hưởng tới 60 % dân
"""

for input_str in list_inputs:
    inputs = tokenizer([input_str])
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    inputs = {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
    }

    encoder_outputs = roberta.encoder(**inputs)
    spoken_tagging_output = roberta.spoken_tagging_classifier(roberta.dropout(encoder_outputs[0]))
    spoken_tagging_output = torch.argmax(spoken_tagging_output, dim=-1).tolist()[0][1:-1] # Skip <bos> and <eos> token label

    # print("Sequence Tagging:", spoken_tagging_output)

    temp = ""
    output_str = ""

    for i, ele in enumerate(input_str.split()):
        if spoken_tagging_output[i] in [0,1] or i == len(input_str.split()) - 1:
            if i == len(input_str.split()) - 1:
                temp += " " + ele
            if len(temp) > 0:
                inputs = tokenizer([temp])
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']

                inputs = {
                    "input_ids": torch.tensor(input_ids),
                    "attention_mask": torch.tensor(attention_mask),
                }

                outputs = roberta.generate(**inputs, output_attentions=True, num_beams=1, num_return_sequences=1)

                for output in outputs.cpu().detach().numpy().tolist():
                    decoded_output = tokenizer.decode(output, skip_special_tokens=True) 
                    output_str += decoded_output if len(output_str) == 0 else " " + decoded_output
            
            if spoken_tagging_output[i] == 0:
                output_str += ele if len(output_str) == 0 else " " + ele
                temp = ""
            else:
                temp = ele
        else:
            temp += ele if len(temp) == 0 else " " + ele

    print("Input:", input_str)
    print("Output:", output_str)
    print()
