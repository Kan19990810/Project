import argparse
import os
import numpy as np
if __name__ == "__main__":
    """
    Adding voxceleb1_root to src_trials_path and getting dst_trials_path
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--voxceleb1_root', help='voxceleb1_root', type=str, default='')
    parser.add_argument('--src_trials_path', help='src_trials_path', type=str, default='lists/veri_test.txt')
    parser.add_argument('--dst_trials_path', help='dst_trials_path', type=str, default='lists/vox1_test.txt')
    args = parser.parse_args()

    trials = np.loadtxt(args.src_trials_path, dtype=str)

    f = open(args.dst_trials_path, 'w')
    for item in trials:
        enroll_path = os.path.join(args.voxceleb1_root, item[1])
        test_path = os.path.join(args.voxceleb1_root, item[2])
        f.write('{} {} {}\n'.format(item[0], enroll_path, test_path))
