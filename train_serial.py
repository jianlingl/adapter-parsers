
import argparse, os
from main import run_train


# 'en', 'de', 'fr', 'he', 'ko', 'sv', 'zh', 'akk', 'kk', 'kmr', 'mr', 'sa', 'ta', 'yo'
# folder = 'data/UD2UC/'
# langs = ['en']
folder = 'data/unify_exist/'
langs = ['fr', 'he', 'ko', 'sv', 'zh', 'akk', 'kk', 'kmr', 'mr', 'sa', 'ta', 'yo']

plm = "/data/hfmodel/bloom-7b1" #'bert-base-multilingual-cased', 'xlm-roberta-large', '/bloom-7b1'
use_adapter = True
use_lang_emb = False
batch_size = 8 # 16 


for lang in langs:
    train_path, dev_path, test_path = folder + lang + '.train', folder + lang + '.dev', folder + lang + '.test'
    flag = 'UD' if 'UD2UC' in folder else 'un'
    file = f"{plm.split('/')[-1].split('-')[0]}_{lang}_{flag}.pt"
    model_path_base = 'log/saved_parsers/mono/' + file

    # assert os.path.exists(test_path), ""
    if not os.path.exists(train_path):
        print(f'no train file in {lang}')
        continue
    if not os.path.exists(dev_path):
        dev_path = train_path

    args = {
        'seed': 20248888, 
        'plm': plm,
        'use_adapter': use_adapter,
        'use_lang_emb': use_lang_emb,
        'batch_size': batch_size,
        'train_path': train_path, 
        'dev_path': dev_path, 
        'test_path': test_path, 
        'model_path_base': model_path_base
        }
    args = argparse.Namespace(**args)
    run_train(args)
