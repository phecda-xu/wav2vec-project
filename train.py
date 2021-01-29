# coding:utf-8
#
import os
import time
import random
import logging
import argparse
import numpy as np
import torch.distributed as dist
import torch.utils.data.distributed
from ASR.data_utils.data_loader import AudioDataLoader, SpectrogramDataset
from ASR.data_utils.data_loader import DSRandomSampler, DSDistributedSampler
from ASR.logger import VisdomLogger, TensorBoardLogger
from ASR.trainer import AMTrainer


def parser_sets():
    parser = argparse.ArgumentParser(description='tdnn training')
    parser.add_argument('--adam', dest='adam', action='store_true', help='Replace SGD with Adam')
    parser.add_argument('--betas', default=(0.9, 0.999), nargs='+', help='ADAM betas')
    parser.add_argument('--batch-size', default=20, type=int, help='Batch size for training')
    parser.add_argument('--best-val-model-name', default='tdnn_final.pth',
                        help='Location to save best validated model within the save folder')
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
    parser.add_argument('--checkpoint', dest='checkpoint', action='store_true',
                        help='Enables checkpoint saving of model')
    parser.add_argument('--checkpoint-per-iteration', default=0, type=int,
                        help='Save checkpoint per iteration. 0 means never save')
    parser.add_argument('--continue-from', default='', help='Continue from checkpoint model')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--distributed', dest='distributed', action='store_true', help='Use cuda to train model')
    parser.add_argument('--decay-epoch', '--decay_epoch', default=5, type=int, help='set weight decay epoch step')
    parser.add_argument('--epochs', default=70, type=int, help='Number of training epochs')
    parser.add_argument('--eps', default=1e-8, type=float, help='ADAM eps')
    parser.add_argument('--finetune', dest='finetune', action='store_true',
                        help='Finetune the model from checkpoint "continue_from"')
    parser.add_argument('--id', default='TDNN training', help='Identifier for visdom/tensorboard run')
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--labels-path', default='Work/*/data/vocab.txt', help='all vocab for transcription')
    parser.add_argument('--local-rank', '--local_rank', default=-1, type=int, help='local rank')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
    parser.add_argument('--lambda-kd', dest='lambda_kd', type=float, default=3000.0)
    parser.add_argument('--log-dir', default='Work/*/visualize/', help='Location of tensorboard log')
    parser.add_argument('--log-params', dest='log_params', action='store_true',
                        help='Log parameter values and gradients')
    parser.add_argument('--learning-anneal', default=1.1, type=float,
                        help='Annealing applied to learning rate every epoch')
    parser.add_argument('--load-auto-checkpoint', dest='load_auto_checkpoint', action='store_true',
                        help='Enable when handling interruptions. Automatically load the latest checkpoint from the '
                             'save folder')
    parser.add_argument('--loss-scale', default=1,
                        help='Loss scaling used by Apex.'
                             ' Default is 1 due to warp-ctc not supporting scaling of gradients')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
    parser.add_argument('--max-durations', default=30.0, type=float, help='max durations of wavfile')
    parser.add_argument('--min-durations', default=0.0, type=float, help='min durations of wavfile')
    parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')
    parser.add_argument('--net-arch', dest='net_arch', type=str, default='base',
                        help="base, super, large, medium or small")
    parser.add_argument('--noise-dir', default=None,
                        help='Directory to inject noise into audio. If default, noise Inject not added')
    parser.add_argument('--noise-prob', default=0.4, type=float, help='Probability of noise being added per sample')
    parser.add_argument('--noise-min', default=0.0,
                        help='Minimum noise level to sample from. (1.0 means all noise, not original signal)',
                        type=float)
    parser.add_argument('--noise-max', default=0.5,
                        help='Maximum noise levels to sample from. Maximum 1.0', type=float)
    parser.add_argument('--opt-level', type=str, default='O0',
                        help='Optimization level to use for training using Apex. Default is FP32 training. '
                             'O1 is mixed precision and recommended for mixed precision hardware')
    parser.add_argument('--reverberation', dest='reverberation', action='store_true',
                        help='add reverberation to audio.')
    parser.add_argument('--reverb-prob', default=0.4, type=float, help='Probability of reverberation added per sample')
    parser.add_argument('--save-n-recent-models', default=0, type=int,
                        help='Maximum number of checkpoints to save. '
                             'If the max is reached, we delete older checkpoints.'
                             'Default is there is no maximum number, so we save all checkpoints.')
    parser.add_argument('--save-folder', default='models/', help='Location to save epoch models')
    parser.add_argument('--speed-volume-perturb', dest='speed_volume_perturb', action='store_true',
                        help='Use random tempo and gain perturbations.')
    parser.add_argument('--seed', default=123456, type=int, help='Seed to generators')
    parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
    parser.add_argument('--teacher-model-path', default='', help='teacher model path')
    parser.add_argument('--train-manifest', metavar='DIR',
                        help='path to train manifest', default='Work/*/data/manifest.*.train')
    parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing')
    parser.add_argument('--visdom', dest='visdom', action='store_true', help='Turn on visdom graphing')
    parser.add_argument('--val-manifest', metavar='DIR',
                        help='path to validation manifest', default='Work/*/data/manifest.*.dev')
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--word-form', dest='word_form', type=str, default='sinogram')
    parser.add_argument('--wd', '--weight_decay', default=1e-5, type=float, help='Initial weight decay')
    parser.add_argument('--vector-model-arch', default='V1', type=str, help='path to load wav2vec model')
    parser.add_argument('--vector-model-path',
                        default='models/V1/wav2vec_large.pt',
                        type=str,
                        help='path to load wav2vec model')
    args = parser.parse_args()
    return args


def main(args):
    # logging INFO
    logging_path = '{}/log'.format(os.path.dirname(os.path.dirname(args.train_manifest)))
    if not os.path.exists(logging_path):
        os.makedirs(logging_path)
    log_name = time.asctime().replace(':', "-").replace(" ", "_")
    logger = logging.getLogger("train.py")
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler("{}/{}_{}_log.txt".format(logging_path, args.net_arch, log_name), mode='w')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # Set seeds for determinism
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    #
    main_proc = True
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        print("Setting CUDA Device to {}".format(args.local_rank))

        dist.init_process_group(backend=args.dist_backend)
        main_proc = args.local_rank == 0  # Main process handles saving of models and reporting
    # 初始化训练器
    asr_trainer = AMTrainer(args)
    # 设置可视化logger工具
    if main_proc and args.visdom:
        visdom_logger = VisdomLogger(args.id, args.epochs)
    else:
        visdom_logger = None
    if main_proc and args.tensorboard:
        log_dir = os.path.join(args.log_dir, 'net_{}'.format(args.net_arch))
        tensorboard_logger = TensorBoardLogger(args.id, log_dir, args.log_params)
    else:
        tensorboard_logger = None
    # 初始化可是工具
    asr_trainer.init_visualizer(visdom_logger=visdom_logger, tensorboard_logger=tensorboard_logger)
    # 调用数据处理
    train_dataset = SpectrogramDataset(audio_conf=asr_trainer.audio_conf,
                                       manifest_filepath=args.train_manifest,
                                       labels=asr_trainer.labels,
                                       word_form=args.word_form,
                                       speed_volume_perturb=args.speed_volume_perturb,
                                       reverberation=args.reverberation,
                                       min_durations=args.min_durations,
                                       max_durations=args.max_durations)
    valid_dataset = SpectrogramDataset(audio_conf=asr_trainer.audio_conf,
                                       manifest_filepath=args.val_manifest,
                                       labels=asr_trainer.labels,
                                       word_form=args.word_form,
                                       speed_volume_perturb=False,
                                       reverberation=False)
    if not args.distributed:
        train_sampler = DSRandomSampler(train_dataset,
                                        batch_size=args.batch_size,
                                        start_index=asr_trainer.state.training_step)
    else:
        train_sampler = DSDistributedSampler(dataset=train_dataset,
                                             batch_size=args.batch_size,
                                             start_index=asr_trainer.state.training_step,
                                             num_replicas=args.world_size,
                                             rank=args.local_rank)
    train_loader = AudioDataLoader(dataset=train_dataset,
                                   num_workers=args.num_workers,
                                   batch_sampler=train_sampler)
    valid_loader = AudioDataLoader(dataset=valid_dataset,
                                   num_workers=args.num_workers,
                                   batch_size=args.batch_size)
    # 打印模型参数量
    if main_proc:
        print(asr_trainer.model)
        print("Number of parameters: %d" % asr_trainer.count_parameter())
    # 训练和验证（train and valid）
    asr_trainer.train(train_loader, valid_loader, main_proc, logger)
    # 最后再次严重保存的最有模型（test best val model in valid dataset again）
    asr_trainer.test(args.best_val_model_name, valid_loader, logger=logger, half=False)


if __name__ == '__main__':
    args = parser_sets()
    print("***************************************")
    print("Load file:")
    for param, value in args.__dict__.items():
        print("{} = {}".format(param, value))
    print("***************************************")
    main(args)
