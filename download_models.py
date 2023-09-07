import os
import argparse
from huggingface_hub import snapshot_download

CACHE_DIR = '.cache'
AVAILABLE_MODELS = ['mkhalifa/grace-discrim-flan-t5-gsm8k']
CKPT_DIR = 'ckpts/discrim/'


def main(args):
    repo_id = 'mkhalifa/grace-discrim-{}-{}'.format(args.lm, args.task)
    assert repo_id in AVAILABLE_MODELS, "Model not available"

    save_dir = os.path.join(CKPT_DIR, repo_id.split('/')[-1]).replace('grace-discrim-', '')
    os.makedirs(save_dir, exist_ok=True)
    print("Downloading {} to {}".format(repo_id, save_dir))
    snapshot_download(
        repo_id=repo_id,
        local_dir_use_symlinks=False,
        local_dir=save_dir,
        cache_dir=CACHE_DIR,
    )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='gsm8k')
    parser.add_argument('--lm', type=str, default='flan-t5', choices=['flan-t5', 'llama-7b', 'llama-13b'])

    args = parser.parse_args()
    main(args)