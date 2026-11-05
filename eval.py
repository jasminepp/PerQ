"""
Evaluate a checkpoint on the datasets of an experiment config (train.py with --eval-only true):

    python eval.py --exp-cfg cfgs/exp/train/perq/mlic_rgb_eval_r64_lamb4-4096_kodak_clic2020-test.yaml \\
        --model-cfg cfgs/model/dcvc-rt_perq/dcvc-rt_intra_perq.yaml \\
        --pretrain <checkpoint> --output-dir outputs --exp-name eval
"""
import sys

from train import main


if __name__ == '__main__':
    if '--eval-only' not in sys.argv:
        sys.argv += ['--eval-only', 'true']
    main()