import time
import os
import argparse
import functools
import torch
import math
import random
import numpy as np
from typing import List

from model.treebank import load_treebank, build_label_vocab, build_tag_vocab, Tree
from model.parser import Parser
from model.learning_rates import WarmupThenReduceLROnPlateau
from model.eval import evalb, eval_ups_lps, FScore

# from config import Hparam
from config_partitioned_transformer import Hparam
from tqdm import tqdm


def get_para_amount(trainable_parameters):
    print("layers of trainable_parameters: ", len(trainable_parameters))
    para_count = 0

    for layer in trainable_parameters:
        layer_count = layer.size()[0]
        if len(layer.size())>1:
            for para in layer.size()[1:]:
                layer_count *= para
        para_count += layer_count

    print('---------'*2, 'para_count', '---------'*2)
    print(para_count)


def printLog(epoch, batch_count, treebank_len, batch_size, batch_loss_value, grad_norm, epoch_start_time, start_time):
    print(
        "epoch {:,} "
        "batch {:,}/{:,} "
        "batch-loss {:.4f} "
        "grad-norm {:.4f} "
        "epoch-elapsed {} "
        "total-elapsed {}".format(
                                epoch,
                                batch_count,
                                int(np.ceil(treebank_len / batch_size)),
                                batch_loss_value,
                                grad_norm,
                                format_elapsed(epoch_start_time),
                                format_elapsed(start_time),
                                )   
        )


def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string


def run_evaluate(parser: Parser, treebank: List[Tree], evalb_dir: str, isTest: bool=False):
    pred_treebank = parser.parse(treebank)
    dev_fscore = evalb(evalb_dir, treebank, pred_treebank)
    
    if isTest:
        label_set = set('::'.join(list(parser.hparam.label_vocab.keys())).split("::"))
        label_set.remove("*")
        _, score_lps = eval_ups_lps(treebank, pred_treebank, label_scores=True, label_set=label_set, len_scores=True)
    else:
        _, score_lps = eval_ups_lps(treebank, pred_treebank)

    # return dev_fscore.fscore  # evalb Fscore 
    return score_lps[2]  # our Fscore 


def save_model(parser: Parser, hparam: Hparam):
    path = hparam.model_path_base + ".pt"
    if os.path.exists(path):
        print("Removing previous model file {}...".format(path))
        os.remove(path)

    torch.save({"hparam": hparam.__dict__, "state_dict": parser.state_dict()}, path)


def run_train(hparam: Hparam):
    np.random.seed(hparam.numpy_seed)
    torch.manual_seed(hparam.numpy_seed)
    torch.cuda.manual_seed_all(hparam.numpy_seed)
    random.seed(hparam.numpy_seed)

    print(hparam.__dict__)

    train_bank = load_treebank(hparam.train_path, max_snt_len=hparam.max_snt_len, top_exist=True)
    dev_bank = load_treebank(hparam.dev_path, max_snt_len=hparam.max_snt_len, top_exist=True)
    test_bank = load_treebank(hparam.test_path, max_snt_len=hparam.max_snt_len, top_exist=True)

    label_vocab = build_label_vocab(train_bank)
    print('---------'*2, 'label_vocab', '---------'*2)
    print(label_vocab)

    tag_vocab = build_tag_vocab(train_bank)
    print('---------'*2, 'tag_vocab', '---------'*2)
    print(tag_vocab)

    hparam.label_vocab = label_vocab
    hparam.tag_vocab = tag_vocab

    # 初始化模型
    print('Initializing model')
    parser = Parser(hparam=hparam, cuda_device=hparam.device)
    parser.to(hparam.device)

    # 初始化数据加载
    data_loader = torch.utils.data.DataLoader(
        train_bank,
        batch_size=hparam.batch_size,
        shuffle=True,
        collate_fn=functools.partial(
            parser.encode_and_collate,
        )
    )

    # 初始化优化器

    trainable_parameters = [param for param in parser.parameters() if param.requires_grad]
    get_para_amount(trainable_parameters)
    optimizer = torch.optim.Adam(trainable_parameters, lr=hparam.learning_rate, betas=(0.9, 0.98), eps=1e-9)
    scheduler = WarmupThenReduceLROnPlateau(
        optimizer, hparam.learning_rate_warmup_steps, mode="max", factor=hparam.step_decay_factor,
        patience=hparam.epoch_decay_patience*(len(train_bank)//hparam.big_batch_size), verbose=True,
    )
    clippable_parameters = trainable_parameters
    grad_clip_threshold = (np.inf if hparam.clip_grad_norm == 0 else hparam.clip_grad_norm)    

    # 训练
    print("Training...")
    check_step = len(train_bank) // (hparam.checks_per_epoch*hparam.batch_size)
    best_dev_fscore, best_test_fscore = -np.inf, -np.inf
    step, batch_loss_value, patience, grad_norm = 0, 0.0, 0, 0.0
    start_time = time.time()
    accm_steps = int(hparam.big_batch_size / hparam.batch_size)

    for epoch in range(hparam.max_epoch):
        epoch_start_time = time.time()
        optimizer.zero_grad()

        for batch_count, batch in enumerate(data_loader):
            step += 1
            parser.train()
            loss = parser.compute_loss(batch)
            loss_value = float(loss.data.cpu().numpy())
            batch_loss_value += loss_value
            if loss_value > 0:
                loss.backward()
            del loss
            
            if step % (check_step // 3) == 0:
                printLog(epoch, batch_count, len(train_bank), hparam.batch_size, batch_loss_value, grad_norm, epoch_start_time, start_time)

            if step % accm_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    clippable_parameters, grad_clip_threshold
                )
                optimizer.step()
                scheduler.step(metrics=best_dev_fscore)
                optimizer.zero_grad()

                # printLog(epoch, batch_count, len(train_bank), hparam.batch_size, batch_loss_value, grad_norm, epoch_start_time, start_time)

                batch_loss_value = 0.0

            if step % check_step == 0:
                print("==========================dev set=========================")
                f_score = run_evaluate(parser, dev_bank, hparam.evalb_dir)
                print("\n")
                print("==========================test set=========================")
                test_f_score = run_evaluate(parser, test_bank, hparam.evalb_dir, isTest=True)
                patience += 1
                if not math.isnan(f_score) and f_score > best_dev_fscore:
                    best_dev_fscore, best_test_fscore = f_score, test_f_score
                    save_model(parser, hparam)
                    print("==========================saving model=========================")
                    patience = 0
                print(f"best f score: {best_dev_fscore}, best test f score: {best_test_fscore}")

        # early stop
        if patience > hparam.early_stop_patience:
            print("Terminating due to lack of improvement in dev fscore.")
            break

    
def run_test(args):
    test_bank = load_treebank(args.test_path, max_snt_len=args.max_snt_len, top_exist=True)
    parser = Parser.from_trained(args.model_path, int(args.device))
    label_set = set('::'.join(list(parser.hparam.label_vocab.keys())).split("::"))
    label_set.remove("*")
    test_pred = parser.parse(test_bank)
    fscore = evalb(args.evalb_dir, test_bank, test_pred)
    # fscore_lps, fscore_ups = eval_ups_lps(test_bank, test_pred, label_scores=True ,label_set=label_set, len_scores=True)
    fscore_lps, fscore_ups = eval_ups_lps(test_bank, test_pred)

        
def run_pred(args):
    pass


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser("train")
    subparser.set_defaults(callback=lambda args: run_train(Hparam(args)))
    subparser.add_argument("--numpy-seed", type=int)
    subparser.add_argument("--device", default='0')
    subparser.add_argument("--train-path", default="")
    subparser.add_argument("--dev-path", default="")
    subparser.add_argument("--test-path", default="")
    subparser.add_argument("--model-path-base", required=True)

    subparser = subparsers.add_parser("test")
    subparser.set_defaults(callback=run_test)
    subparser.add_argument("--device", default='0')
    subparser.add_argument("--model-path", required=True)
    subparser.add_argument("--test-path", default="")
    subparser.add_argument("--evalb-dir", default="EVALB")
    subparser.add_argument("--max-snt-len", default=160)

    subparser = subparsers.add_parser("pred")
    subparser.set_defaults(callback=run_pred)
    subparser.add_argument("--device", default='0')
    subparser.add_argument("--model-path", required=True)
    subparser.add_argument("--pred-raw", default="")
    subparser.add_argument("--opt-path", default="dataset/pred_4annotation_projection")

    args = parser.parse_args()
    args.callback(args)

if __name__ == "__main__":
    main()




        


