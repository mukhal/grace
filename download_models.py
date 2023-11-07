import os
import argparse
from huggingface_hub import snapshot_download

CACHE_DIR = '.cache'
AVAILABLE_MODELS = ['mkhalifa/grace-discrim-gsm8k', 
                    'mkhalifa/grace-discrim-svamp',
                    'mkhalifa/grace-discrim-multiarith', 
                    'mkhalifa/grace-discrim-mathqa',
                    'mkhalifa/grace-discrim-tso',
                    'mkhalifa/grace-discrim-coin_flip']

CKPT_DIR = 'ckpts/discrim/'


def main(args):
    repo_id = 'mkhalifa/grace-discrim-{}'.format(args.task)
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

    args = parser.parse_args()
    main(args)