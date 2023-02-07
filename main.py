from transformers import Seq2SeqTrainingArguments
import os
import model
import preprocess
from my_trainer import InvertTextNormalizationTrainer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    roberta, tokenizer = model.init_model()

    dataset = preprocess.init_data()

    data_collator = preprocess.DataCollatorInvertTextNormalization(tokenizer, model=roberta)

    num_epochs = 15
    checkpoint_path = "./checkpoints"
    batch_size = 8
    """
    eval_accumulation_steps (int, optional): 
            Number of predictions steps to accumulate the output tensors for, before moving the results to the 
            CPU. If left unset, the whole predictions are accumulated on GPU/TPU before being moved to 
            the CPU (faster but requires more memory).

    save_steps (`int`, *optional*, defaults to 500):
            Number of updates steps before two checkpoint saves if `save_strategy="steps"`.
    """
    training_args = Seq2SeqTrainingArguments(
        output_dir=checkpoint_path,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=2e-5,
        gradient_accumulation_steps=1,
        predict_with_generate=True,
        # save_total_limit=2,
        do_train=True,
        do_eval=True,
        logging_steps=5000,
        save_steps=18740,
        eval_steps=5000,
        num_train_epochs=num_epochs,
        warmup_ratio=1/num_epochs,
        logging_dir=os.path.join(checkpoint_path,'log'),
        overwrite_output_dir=True,
        eval_accumulation_steps=10,
        dataloader_num_workers=0,
        generation_max_length=50,
        fp16=True,
        ignore_data_skip=True
    )

    trainer = InvertTextNormalizationTrainer(
        model=roberta,
        args=training_args,
        train_dataset=dataset['train'].shard(50, 0),
        eval_dataset=dataset['valid'],
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    trainer.train()
    # trainer.evaluate()
    # trainer.save_model(checkpoint_path)
    # trainer.evaluate()