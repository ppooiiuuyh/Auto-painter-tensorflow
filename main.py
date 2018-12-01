from autopainter import AutoPainter
import argparse
from utils import *
from generator import Generator
from discriminator import Discriminator

# ======================================================================================================================
# Parsing and Configuration
# ======================================================================================================================
def parse_args():
    desc = "Tensorflow implementation of AutoPainter"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--model_name", type=str, default="AutoPainter")
    parser.add_argument("--gpu", type=int, default=0)  # -1 for CPU
    parser.add_argument('--phase', type=str, default='train', help='train or test ?')
    parser.add_argument('--dataset', type=str, default='line_art', help='dataset_name') #mapssmall, edges2shoessamll,
    parser.add_argument('--path_trainA', type=str, default='./../datasets/autocolorization/illustrations_resized_256/xdog/', help='dataset_name')
    parser.add_argument('--path_trainB', type=str, default='./../datasets/autocolorization/illustrations_resized_256/original/', help='dataset_name')
    parser.add_argument('--path_testA', type=str, default='./../datasets/autocolorization/lineart/', help='dataset_name')

    parser.add_argument('--epoch', type=int, default=200, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch per gpu')
    parser.add_argument('--print_freq', type=int, default=100, help='The number of image_print_freq')
    parser.add_argument('--lr', type=float, default=0.0002, help='The learning rate')
    parser.add_argument('--L1_weight', type=float, default=100.0, help='The L1 lambda')
    parser.add_argument('--FL_weight', type=float, default=0.01, help='The FL lambda')
    parser.add_argument('--TV_weight', type=float, default=0.0001, help='The TV lambda')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--repeat', type=int, default=9, help='img size : 256 -> 9, 128 -> 6')
    parser.add_argument('--img_size', type=int, default=512, help='The size of image')
    parser.add_argument('--gray_to_RGB', type=bool, default=False, help='Gray -> RGB')

    """ generator """
    parser.add_argument('--model_name_gen', type=str, default='gen')
    parser.add_argument('--checkpoint_dir_gen', type=str, default='checkpoint_gen')
    parser.add_argument('--checkpoint_itr_gen', type=int, default=0)


    """ discriminator """
    parser.add_argument('--model_name_dis', type=str, default='dis')
    parser.add_argument('--checkpoint_dir_dis', type=str, default='checkpoint_dis')
    parser.add_argument('--checkpoint_itr_dis', type=int, default=0)

    parser.add_argument('--result_dir', type=str, default='results')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--sample_dir', type=str, default='samples')

    return check_args(parser.parse_args())


"""checking arguments"""
# ======================================================================================================================
# Checking arguments
# ======================================================================================================================
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir_gen)
    check_folder(args.checkpoint_dir_dis)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --epoch
    try:    assert args.epoch >= 1
    except: print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:    assert args.batch_size >= 1
    except: print('batch size must be larger than or equal to one')
    return args

# ======================================================================================================================
# Main
# ======================================================================================================================
def main():
    args = parse_args()
    if args is None: exit()

    """system configuration"""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    config = tf.ConfigProto()
    if args.gpu == -1: config.device_count = {'GPU': 0}
    #config.gpu_options.per_process_gpu_memory_fraction = 0.3
    config.allow_soft_placement = True #auto?
    # config.operation_timeout_in_ms=10000


    """system configuration"""
    with tf.Session(config=config) as sess:
        autopainter = AutoPainter(sess = sess, args=args)
        
        if args.phase == 'train' :
            autopainter.train()
            print(" [*] Training finished!")

        elif args.phase == 'test' :
            autopainter.test()
            print(" [*] Test finished!")
        
if __name__ == '__main__':
    main()