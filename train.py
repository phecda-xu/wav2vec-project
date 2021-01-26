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
from ASR.model_utils.model import TDNNet, FCFintune
from ASR.model_utils.model import load_wav2vec_model
from ASR.model_utils.state import TrainingState
from ASR.model_utils.utils import CheckpointHandler, check_loss
from ASR.model_utils.decoder import GreedyDecoder
from test import evaluate

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel
    from apex.parallel import convert_syncbn_model
    apex_state = True
except:
    print("APEX: Cannot use apex, loading torch DistributedDataParallel instead")
    from torch.nn.parallel import DistributedDataParallel
    apex_state = False

parser = argparse.ArgumentParser(description='tdnn training')
parser.add_argument('--adam', dest='adam', action='store_true', help='Replace SGD with Adam')
parser.add_argument('--betas', default=(0.9, 0.999), nargs='+', help='ADAM betas')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for training')
parser.add_argument('--best-val-model-name', default='tdnn_final.pth',
                    help='Location to save best validated model within the save folder')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--checkpoint-per-iteration', default=0, type=int,
                    help='Save checkpoint per iteration. 0 means never save')
parser.add_argument('--continue-from', default='', help='Continue from checkpoint model')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--distributed', dest='distributed', action='store_true', help='Use cuda to train model')
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
parser.add_argument('--log-dir', default='visualize/tdnn_final', help='Location of tensorboard log')
parser.add_argument('--log-params', dest='log_params', action='store_true', help='Log parameter values and gradients')
parser.add_argument('--learning-anneal', default=1.1, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--load-auto-checkpoint', dest='load_auto_checkpoint', action='store_true',
                    help='Enable when handling interruptions. Automatically load the latest checkpoint from the '
                    'save folder')
parser.add_argument('--loss-scale', default=1,
                    help='Loss scaling used by Apex. Default is 1 due to warp-ctc not supporting scaling of gradients')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--max-durations', default=30.0, type=float, help='max durations of wavfile')
parser.add_argument('--min-durations', default=0.0, type=float, help='min durations of wavfile')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--net-arch', dest='net_arch', type=str, default='base', help="base, super, large, medium or small")
parser.add_argument('--noise-dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise-prob', default=0.4, type=float, help='Probability of noise being added per sample')
parser.add_argument('--noise-min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise-max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--opt-level', type=str, default='O0',
                    help='Optimization level to use for training using Apex. Default is FP32 training. '
                         'O1 is mixed precision and recommended for mixed precision hardware')
parser.add_argument('--reverberation', dest='reverberation', action='store_true',
                    help='add reverberation to audio.')
parser.add_argument('--reverb-prob', default=0.4, type=float, help='Probability of reverberation added per sample')
parser.add_argument('--save-n-recent-models', default=0, type=int,
                    help='Maximum number of checkpoints to save. If the max is reached, we delete older checkpoints.'
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


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    args = parser.parse_args()
    #
    print("***************************************")
    print("Load file:")
    for param, value in args.__dict__.items():
        print("{} = {}".format(param, value))
    print("***************************************")
    # logging
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

    main_proc = True
    device = torch.device("cuda" if args.cuda else "cpu")

    if args.distributed:
        # when using NCCL, on failures, surviving nodes will deadlock on NCCL ops
        # because NCCL uses a spin-lock on the device. Set this env var and
        # to enable a watchdog thread that will destroy stale NCCL communicators
        torch.cuda.set_device(args.local_rank)
        print("Setting CUDA Device to {}".format(args.local_rank))

        dist.init_process_group(backend=args.dist_backend)
        main_proc = args.local_rank == 0  # Main process handles saving of models and reporting

    checkpoint_handler = CheckpointHandler(save_folder=args.save_folder,
                                           best_val_model_name=args.best_val_model_name,
                                           checkpoint_per_iteration=args.checkpoint_per_iteration,
                                           save_n_recent_models=args.save_n_recent_models)
    # if main_proc and args.visdom:
    #     visdom_logger = VisdomLogger(args.id, args.epochs)
    # if main_proc and args.tensorboard:
    #     tensorboard_logger = TensorBoardLogger(args.id, args.log_dir, args.log_params)

    # avg_loss, start_epoch, start_iter, optim_state, amp_state = 0, 0, 0, None, None
    if args.load_auto_checkpoint:
        latest_checkpoint = checkpoint_handler.find_latest_checkpoint()
        if latest_checkpoint:
            args.continue_from = latest_checkpoint

    if args.continue_from:  # Starting from previous model
        state = TrainingState.load_state(state_path=args.continue_from, network=TDNNet)
        state.init_results_tracking(epochs=args.epochs)
        model = state.model
        if args.finetune:
            state.init_finetune_states(args.epochs)

        # if main_proc and args.visdom:  # Add previous scores to visdom graph
        #     visdom_logger.load_previous_values(state.epoch, state.results)
        # if main_proc and args.tensorboard:  # Previous scores to tensorboard logs
        #     tensorboard_logger.load_previous_values(state.epoch, state.results)
    else:
        # Initialise new model training
        with open(args.labels_path) as label_file:
            # labels = str(''.join(json.load(label_file)))
            labels = [i.strip('\n') for i in label_file.readlines()]
            if ' ' not in labels:
                labels.append(' ')

        audio_conf = dict(sample_rate=args.sample_rate,
                          noise_dir=args.noise_dir,
                          noise_prob=args.noise_prob,
                          noise_levels=(args.noise_min, args.noise_max),
                          reverb_prob=args.reverb_prob,
                          net_arch=args.net_arch)
        if args.vector_model_arch == "V1":
            input_channels = 512
        else:
            input_channels = 768
        if args.net_arch == "base":
            model = FCFintune(labels=labels, audio_conf=audio_conf, input_channels=input_channels)
        else:
            model = TDNNet(labels=labels, audio_conf=audio_conf, input_channels=input_channels)
        state = TrainingState(model=model)
        state.init_results_tracking(epochs=args.epochs)
        #
    # teacher model
    vec_model = load_wav2vec_model(args.vector_model_path)
    vec_model.eval()
    # Data setup
    evaluation_decoder = GreedyDecoder(model.labels, word_form=args.word_form)  # Decoder used for validation
    train_dataset = SpectrogramDataset(audio_conf=model.audio_conf,
                                       manifest_filepath=args.train_manifest,
                                       labels=model.labels,
                                       word_form=args.word_form,
                                       speed_volume_perturb=args.speed_volume_perturb,
                                       reverberation=args.reverberation,
                                       min_durations=args.min_durations,
                                       max_durations=args.max_durations)
    test_dataset = SpectrogramDataset(audio_conf=model.audio_conf,
                                      manifest_filepath=args.val_manifest,
                                      labels=model.labels,
                                      word_form=args.word_form,
                                      speed_volume_perturb=False,
                                      reverberation=False)
    if not args.distributed:
        train_sampler = DSRandomSampler(train_dataset,
                                        batch_size=args.batch_size,
                                        start_index=state.training_step)
    else:
        train_sampler = DSDistributedSampler(dataset=train_dataset,
                                             batch_size=args.batch_size,
                                             start_index=state.training_step,
                                             num_replicas=args.world_size,
                                             rank=args.local_rank)
    train_loader = AudioDataLoader(dataset=train_dataset,
                                   num_workers=args.num_workers,
                                   batch_sampler=train_sampler)
    test_loader = AudioDataLoader(dataset=test_dataset,
                                  num_workers=args.num_workers,
                                  batch_size=args.batch_size)

    if args.cuda and args.distributed and apex_state:
        model = convert_syncbn_model(model).to(device)
    else:
        model = model.to(device)
    parameters = model.parameters()

    if args.adam:
        optimizer = torch.optim.AdamW(parameters,
                                      lr=args.lr,
                                      betas=args.betas,
                                      eps=args.eps,
                                      weight_decay=args.wd)
    else:
        optimizer = torch.optim.SGD(parameters,
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    nesterov=True,
                                    weight_decay=args.wd)

    if args.cuda and apex_state:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.opt_level,
                                          loss_scale=args.loss_scale)
        state.track_optim_state(optimizer)
        state.track_amp_state(amp)

    if args.cuda and args.distributed:
        if apex_state:
            model = DistributedDataParallel(model)
        else:
            model = DistributedDataParallel(model,
                                            device_ids=[args.local_rank],
                                            output_device=args.local_rank,
                                            find_unused_parameters=True)
    if main_proc:
        print(model)
        print("Number of parameters: %d" % TDNNet.get_param_size(model))

    criterion = torch.nn.CTCLoss()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    ctc_losses = AverageMeter()

    for epoch in range(state.epoch, args.epochs):
        model.train()
        end_time = time.time()
        start_epoch_time = time.time()
        state.set_epoch(epoch=epoch)
        train_sampler.set_epoch(epoch=epoch)
        train_sampler.reset_training_step(training_step=state.training_step)
        for i, (data) in enumerate(train_loader, start=state.training_step):
            state.set_training_step(training_step=i)
            inputs, targets, input_percentages, target_sizes = data
            # measure data loading time
            data_time.update(time.time() - end_time)
            inputs = inputs.to(device)
            # wav2vec
            if args.vector_model_arch == "V1":
                z = vec_model.feature_extractor(inputs)
                c = vec_model.feature_aggregator(z)
            elif args.vector_model_arch == "V2":
                c = vec_model(inputs, mask=False, features_only=True)["x"]
                c = c.transpose(1, 2)
            else:
                raise ValueError("wrong type of vector_model_arch:{}".format(args.vector_model_arch))
            #
            input_sizes = input_percentages.mul(int(c.size(2))).int()
            #
            out, output_sizes, _ = model(c, input_sizes)
            # ctc_loss:
            out = out.transpose(0, 1)  # TxNxH
            out = out.log_softmax(-1)
            float_out = out.float()  # ensure float32 for loss
            loss = criterion(float_out,
                             targets.to(device).long(),
                             output_sizes.to(device),
                             target_sizes.to(device))
            loss_value = loss.item()

            # Check to ensure valid loss was calculated
            valid_loss, error = check_loss(loss, loss_value)

            del target_sizes, targets, out, float_out

            if valid_loss:
                optimizer.zero_grad()
                # compute gradient
                if args.cuda and apex_state:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
                else:
                    loss.backward()
                optimizer.step()
            else:
                print(error)
                print('Skipping grad update')
                loss_value = 0

            state.avg_loss += loss_value
            losses.update(loss_value, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end_time)
            end_time = time.time()
            if main_proc:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       (epoch + 1), (i + 1), len(train_loader), batch_time=batch_time, data_time=data_time,
                       loss=losses))

            if main_proc and args.checkpoint_per_iteration:
                checkpoint_handler.save_iter_checkpoint_model(epoch=epoch, i=i, state=state)

            del loss,input_sizes, inputs
            # 清理显存占用
            torch.cuda.empty_cache()

        state.avg_loss /= len(train_loader)

        epoch_time = time.time() - start_epoch_time
        if main_proc:
            print('Training Summary Epoch: [{0}]\t'
                  'Time taken (s): {epoch_time:.0f}\t'
                  'Average Loss {loss:.3f}\t'.format(epoch + 1, epoch_time=epoch_time, loss=state.avg_loss))

        with torch.no_grad():
            wer, cer, output_data = evaluate(test_loader=test_loader,
                                             device=device,
                                             model=model,
                                             decoder=evaluation_decoder,
                                             target_decoder=evaluation_decoder)
        state.add_results(epoch=epoch,
                          loss_result=state.avg_loss,
                          wer_result=wer,
                          cer_result=cer)
        if main_proc:
            print('Validation Summary Epoch: [{0}]\t'
                  'Average WER {wer:.3f}\t'
                  'Average CER {cer:.3f}\t'.format(epoch + 1, wer=wer, cer=cer))
            logger.info('Validation Summary Epoch: [{0}]\t'
                        'Average WER {wer:.3f}\t'
                        'Average CER {cer:.3f}\t'.format(epoch + 1, wer=wer, cer=cer))

        # if main_proc and args.visdom:
        #     visdom_logger.update(epoch, state.result_state)
        # if main_proc and args.tensorboard:
        #     tensorboard_logger.update(epoch, state.result_state, model.named_parameters())

        if main_proc and args.checkpoint:  # Save epoch checkpoint
            checkpoint_handler.save_checkpoint_model(epoch=epoch, state=state)
        # anneal lr:
        if main_proc and (epoch + 1) % 5 == 0:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] / args.learning_anneal
            print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))

        if main_proc and (state.best_wer is None or state.best_wer > cer):
            checkpoint_handler.save_best_model(epoch=epoch, state=state)
            state.set_best_wer(cer)
            state.reset_avg_loss()
        state.reset_training_step()  # Reset training step for next epoch
