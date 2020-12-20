# coding=utf-8
from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import glob

import numpy as np
import math
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              WeightedRandomSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from model import BertForRelationExtraction

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig, BertForQuestionAnswering,
                          BertTokenizer)
from transformers import AdamW, WarmupLinearSchedule

from utils_nyt import read_nyt_examples, convert_examples_to_features
from modeling_utils import min_ce_loss, ce_loss, dsloss, f_measure
from utils_nyt import write_nyt_predictions, RawResult

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys())
                  for conf in (BertConfig,)), ())

MODEL_CLASSES = {
    'nyt': (BertConfig, BertForRelationExtraction, BertTokenizer),
    'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer)
}

LOSS_FUNC = {
    'min': min_ce_loss,
    'mean': ce_loss,
    'dsloss': dsloss
}

F_BETA = math.sqrt(0.5)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def nyt_train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    if args.loss_func in LOSS_FUNC:
        loss_func = LOSS_FUNC[args.loss_func]
        print('using %s' % args.loss_func)
        risk_sensitive = args.risk_sensitive
    else:
        loss_func = None
        risk_sensitive = False
    train_dataset, weights = train_dataset

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    if args.max_count > 0:
        train_sampler = WeightedRandomSampler(weights, args.max_count,
                                              replacement=False)
    else:
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // \
            (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // \
            args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    warmup_steps = min(1000, int(args.warmup_propotion * t_total))
    scheduler = WarmupLinearSchedule(optimizer,
                                     warmup_steps=warmup_steps,
                                     t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = DistributedDataParallel(model, device_ids=[args.local_rank],
                                        output_device=args.local_rank,
                                        find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Warmup steps = %d", warmup_steps)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch",
                            disable=args.local_rank not in [-1, 0])
    set_seed(args)  # For reproductibility (even between python 2 and 3)
    for train_it in train_iterator:
        epoch_iterator = tqdm(train_dataloader,
                              desc="Iteration",
                              disable=args.local_rank not in [-1, 0])
        # P, PP, TP = 0, 0, 0
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'candidate_index': batch[1],
                      'candidate_length': batch[2],
                      'attention_mask': batch[3],
                      'entity_type_ids': batch[5],
                      'answer_mask': batch[6]}
            outputs = model(**inputs, loss_func=loss_func,
                            risk_sensitive=risk_sensitive,
                            lambda_weight=args.lambda_weight,
                            gamma=args.gamma)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                clip_grad_norm_(amp.master_params(optimizer),
                                args.max_grad_norm)
            else:
                loss.backward()
                clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logger.info('Training loss @ step %d is %.5f' % (global_step, (tr_loss - logging_loss)/args.logging_steps) )
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = nyt_evaluate(args, model, tokenizer, prefix='ep_' + str(global_step))
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.local_rank in [-1, 0] and (args.save_steps <= 0 or global_step % args.save_steps != 0):
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, 'checkpoint-epoch-{}'.format(train_it))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)

        if args.local_rank == -1:
            results = nyt_evaluate(args, model, tokenizer, prefix=str(train_it))
            for key, value in results.items():
                tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def nyt_evaluate(args, model, tokenizer, prefix=""):
    dataset, features, examples = load_and_cache_nyt_examples(args, tokenizer, evaluate=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    results = []
    all_results = []
    answer_masks = []
    answer_scores = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':       batch[0],
                      'candidate_index':      batch[1],
                      'candidate_length':  batch[2],
                      'attention_mask':  batch[3],
                      'token_type_ids': batch[4],
                      'entity_type_ids': batch[5]}
            outputs = model(**inputs)

            answer_mask = batch[6].bool().float()
            answer_masks.append(answer_mask)
            answer_scores.append(F.softmax(outputs[0], dim=-1))
            log_probs, predictions = outputs[0].max(dim=-1)
            assert log_probs.size(0) == answer_mask.size(0)
            for log_prob, prediction, answer in zip(log_probs, predictions, answer_mask):
                golden = None
                pred = None
                prob = None
                if answer[0] > 0:
                    golden = 0
                else:
                    golden = 1
                if prediction <= 0:
                    pred = 0
                elif answer[prediction] > 0:
                    pred = 1
                else:
                    pred = 2
                prob = log_prob.item()
                results.append((golden, pred, prob))
            example_indices = batch[9]
        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            result = RawResult(unique_id=unique_id,
                               logits=to_list(outputs[0][i]))
            all_results.append(result)
    all_p = len([r for r in results if r[0] > 0])
    all_pp = sorted([r for r in results if r[1] > 0],
                    key=lambda x: x[2], reverse=True)
    all_tp = len(list(filter(lambda x: x[0] == x[1], all_pp)))

    p_all = all_tp / (len(all_pp) if len(all_pp) > 0 else 1.)
    r_all = all_tp / (float(all_p) if all_p > 0 else 1.)
    f1_all = f_measure(p_all, r_all, F_BETA)
    print_results = {
        'all_precision': p_all,
        'all_recall': r_all,
        'all_f1': f1_all,
        'total_positive': all_p,
        'predicted_positive': len(all_pp)
    }
    for num in (10, 20, 50, 100, 200, 300, 400, 500):
        if num > len(all_pp):
            break
        t_pp = all_pp[:num]
        t_tp = len(list(filter(lambda x: x[0] == x[1], t_pp)))
        p_n = t_tp / num
        print_results['p@%d' % num] = p_n

    print(print_results)

    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))

    write_nyt_predictions(examples, features, all_results, output_prediction_file, output_nbest_file)

    return print_results


def load_and_cache_nyt_examples(args, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_file = args.predict_file if evaluate else args.train_file
    input_dir, input_file_name = os.path.split(input_file)
    input_file_text, _ = os.path.splitext(input_file_name)
    cached_features_file = os.path.join(input_dir, 'cached_{}_{}'.format(
        input_file_text,
        str(args.max_seq_length)))

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cache: %s", cached_features_file)

        if evaluate:
            features, examples = torch.load(cached_features_file)
        else:
            features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", input_file)
        examples = read_nyt_examples(input_file=input_file,
                                     is_training=not evaluate,
                                     is_with_negative=args.is_with_negative)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                doc_stride=args.doc_stride,
                                                max_query_length=args.max_query_length,
                                                is_training=not evaluate,
                                                retain_entity=args.retain_entity,)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            if evaluate:
                torch.save((features, examples), cached_features_file)
            else:
                torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    dataset = build_dataset(features, evaluate=evaluate)

    if evaluate:
        return dataset, features, examples
    else:
        if args.max_count > 0:
            nil_count = 0
            total_count = len(features)
            for f in features:
                if f.is_impossible:
                    nil_count += 1
            pos_count = total_count - nil_count
            n_weight = (args.max_count - pos_count) / nil_count
            weights = [n_weight if f.is_impossible else 1. for f in features]
        else:
            weights = None
    return dataset, weights

def build_dataset(features, evaluate=False):
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_candidate_index = torch.tensor([f.candidate_index for f in features], dtype=torch.long)
    all_candidate_length = torch.tensor([f.candidate_length for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    all_entity_type_ids = torch.tensor([f.entity_type_ids for f in features], dtype=torch.long)
    all_answer_mask = torch.tensor([f.answer_mask for f in features], dtype=torch.long)
    if evaluate:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_candidate_index, all_candidate_length,
                                all_input_mask, all_segment_ids, all_entity_type_ids,
                                all_answer_mask, all_cls_index, all_p_mask, all_example_index)
    else:
        dataset = TensorDataset(all_input_ids, all_candidate_index, all_candidate_length,
                                all_input_mask, all_segment_ids, all_entity_type_ids,
                                all_answer_mask, all_cls_index, all_p_mask)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="training data")
    parser.add_argument("--predict_file", default=None, type=str, required=True,
                        help="evaluation data")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, # required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument('--is_with_negative', action='store_true',
                        help='If true, there are negative questions')
    parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--retain_entity", action='store_true',
                        help="Set this flag if you want to retain entity untokenized.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_propotion", default=0., type=float,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--loss_func", default=None,
                        help="loss function used to learn:min, mean, dsloss or None")
    parser.add_argument("--risk_sensitive", action='store_true',
                        help="whether to use adaptive weight")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--max_count", type=int, default=-1,
                        help="Whether or not to keep balance")
    parser.add_argument("--lambda_weight", default=0.5, type=float,
                        help="lambda weight for the noise-tolerant part")
    parser.add_argument("--gamma", default=1.0, type=float,
                        help="gamma scale for the risk-sensitive factor")

    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=-1,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    args = parser.parse_args()

    if args.output_dir:
        if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
            raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
    else:
        assert not args.do_train

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES['nyt']
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_nyt_examples(args, tokenizer, evaluate=False)
        global_step, tr_loss = nyt_train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Save the trained model and the tokenizer
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            # Create output directory if needed
            if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
                os.makedirs(args.output_dir)

            logger.info("Saving model checkpoint to %s", args.output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    if args.do_eval and args.local_rank in [-1, 0]:
        if args.output_dir:
            checkpoints = [args.output_dir]
            logger.info("Evaluate the following checkpoints: %s", checkpoints)
            for checkpoint in checkpoints:
                # Reload the model
                global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
                model = model_class.from_pretrained(checkpoint)
                model.to(args.device)

                # Evaluate
                if args.do_train:
                    eval_prefix = 'dev'
                else:
                    eval_prefix = 'test'
                result = nyt_evaluate(args, model, tokenizer, prefix=eval_prefix)
                result = dict((k + ('_{}'.format(global_step) if global_step else ''), v) for k, v in result.items())
        else:
            model.to(args.device)
            result = nyt_evaluate(args, model, tokenizer, prefix='test')


if __name__ == "__main__":
    main()
