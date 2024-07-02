
import argparse, os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from main import run_train


folder = 'data/UD2UC/'
langs = ['en', 'de', 'fr', 'he', 'ko', 'sv', 'zh', 'akk', 'kk', 'kmr', 'mr', 'sa', 'ta', 'yo']

for lang in langs:
    train_path, dev_path, test_path = folder + lang + '.train', folder + lang + '.dev', folder + lang + '.test'
    model_path_base = 'log/saved_parsers/' + lang + '_xlmr.pt'

    assert os.path.exists(test_path), ""

    if not os.path.exists(train_path):
        print(f'no train file in {lang}')
        continue

    if not os.path.exists(dev_path):
        dev_path = train_path

    args = {'device': '4', 'seed': 20248888, 'train_path': train_path, 'dev_path': dev_path, 'test_path': test_path, 'model_path_base': 'log/saved_parsers/all_mul_mbert_LoRA.pt'}
    args = argparse.Namespace(**args)
    run_train(args)
