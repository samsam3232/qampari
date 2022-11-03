import argparse
import json

import torch
import random
from src.model import FiDT5
import src.data
import src.util
from torch.utils.data import DataLoader, RandomSampler
import transformers
from data.fid_collator import FiD_Collator
from data.fid_dataset import FiD_Data
from data.cb_dataset import CB_Data
from data.cb_collator import CB_Collator


def evaluate(model, dataset, dataloader, tokenizer, answer_max_tokens, output_path, use_fid):
    model.eval()
    answers = list()
    model = model.module if hasattr(model, "module") else model

    with torch.no_grad():
        for i, batch in enumerate(dataloader):

            logger.info(f'In eval: {i}')

            (_, idx, _, _, context_ids, context_mask) = batch
            if use_fid:
                outputs = model.generate(input_ids=context_ids.cuda(), attention_mask=context_mask.cuda(),
                                         max_length=answer_max_tokens)
            else:
                outputs = model.generate(input_ids=context_ids.cuda(), max_length=answer_max_tokens)
            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=False)
                data_sample = dataset.get_example(idx[k])
                data_sample['fid_pred'] = ans
                answers.append(data_sample)

            if random.random() < 0.001:
                print(dataset.get_example(idx[k])['question'])
                print(dataset.get_example(idx[k])['answers'])
                print(ans)

            if i % 10 == 0:
                with open(output_path.replace('jsonl', 'txt'), 'w') as f:
                    f.write(str(i))

    with open(output_path, 'w') as f:
        for sample in answers:
            f.write(json.dumps(sample) + '\n')


def main(input_path: str, checkpoint_path: str, cache_dir: str, text_maxlength: int, answer_maxlengths: int,
         per_gpu_batch_size: int, num_workers: int, output_dir: str, logger_path: str, model_size: str = 'base',
         num_contexts: int = 50,
         use_fid: str = "true"):
    examples = src.data.load_data(input_path)
    dataset = FiD_Data(examples, num_contexts)
    print('After loading data')

    model_name = 't5-' + model_size
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    print('After loading tokenizer')

    if use_fid == "true":
        print('Before loading FiD')
        model = FiDT5.from_pretrained(checkpoint_path)
        print('After loading FiD')
    else:
        model = transformers.T5ForConditionalGeneration.from_pretrained(checkpoint_path)
    model = model.cuda()
    print('After passing to cuda')

    Collator = FiD_Collator if use_fid == "true" else CB_Collator

    collator = Collator(text_maxlength, tokenizer, answer_maxlength=answer_maxlengths)
    sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=sampler, batch_size=per_gpu_batch_size, drop_last=False,
                                  collate_fn=collator, num_workers=num_workers)

    #    print('After passing model to cuda')
    evaluate(model, dataset, train_dataloader, tokenizer, answer_maxlengths, output_dir, use_fid == "true")
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser("FiD evaluation")
    parser.add_argument('-i', '--input_path', type=str, help='Path to where the data we want to evaluate')
    parser.add_argument('-c', '--checkpoint_path', type=str, help="Help to where the models are kept")
    parser.add_argument('-n', '--num_contexts', type=int, default=50,
                        help='How many contexts we can see in the question')
    parser.add_argument('-s', '--model_size', type=str, default='base', help='Size of the t5 model')
    parser.add_argument('-p', '--per_gpu_batch_size', type=int, default=1, help='Elements per GPU')
    parser.add_argument('-l', '--logger_path', type=str)
    parser.add_argument('-w', '--num_workers', type=int, default=2, help="Workers to retrieve the data")
    parser.add_argument('-o', '--output_dir', type=str, help="Path to where to keep the results")
    parser.add_argument('-u', '--use_fid', type=str, default='true', help="Whether or not to use FiD based models")
    parser.add_argument('-t', '--text_maxlength', type=int, default=256)
    parser.add_argument('-a', '--answer_maxlengths', type=int, default=100)
    parser.add_argument('--cache_dir', type=str, default='./hf_model')
    args = parser.parse_args()

    logger = src.util.init_logger(filename=args.logger_path)
    main(**vars(args))
