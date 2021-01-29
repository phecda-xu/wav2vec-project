# coding:utf-8
#
#
import time
import torch
from tqdm import tqdm
from ASR.model_utils.model import TDNNet, FCFintune
from ASR.model_utils.model import load_wav2vec_model
from ASR.model_utils.state import TrainingState
from ASR.model_utils.utils import CheckpointHandler, check_loss, load_model
from ASR.model_utils.decoder import GreedyDecoder
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel
    from apex.parallel import convert_syncbn_model
    apex_state = True
except:
    print("APEX: Cannot use apex, loading torch DistributedDataParallel instead")
    from torch.nn.parallel import DistributedDataParallel
    apex_state = False


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


class AMTrainer(object):
    def __init__(self, args):
        self.criterion = torch.nn.CTCLoss()
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.losses = AverageMeter()
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.args = args
        self.audio_conf = dict(sample_rate=args.sample_rate,
                               noise_dir=args.noise_dir,
                               noise_prob=args.noise_prob,
                               noise_levels=(args.noise_min, args.noise_max),
                               reverb_prob=args.reverb_prob)
        self.model_conf = dict(net_arch=args.net_arch)
        self.net_arch = args.net_arch
        self.labels = self.load_labels(args.labels_path)
        #
        self.checkpoint_handler = CheckpointHandler(save_folder=args.save_folder,
                                                    best_val_model_name=args.best_val_model_name,
                                                    checkpoint_per_iteration=args.checkpoint_per_iteration,
                                                    save_n_recent_models=args.save_n_recent_models)
        #
        self.vector_model_arch = args.vector_model_arch
        if self.net_arch == "base":
            self.net_work = FCFintune
        else:
            self.net_work = TDNNet
        # prepare state and model
        self._init_state_and_model()
        self._wav2vec_model()
        # prepare optimizer and device
        self._init_optimizer_and_device()
        #
        self.decoder = GreedyDecoder(self.labels, word_form=args.word_form)

    def _build_model(self):
        if self.vector_model_arch == "V1":
            input_channels = 512
        else:
            input_channels = 768
        self.model = self.net_work(net_arch=self.net_arch,
                                   labels=self.labels,
                                   audio_conf=self.audio_conf,
                                   input_channels=input_channels)

    def _init_state_and_model(self):
        if self.args.load_auto_checkpoint:
            latest_checkpoint = self.checkpoint_handler.find_latest_checkpoint()
            if latest_checkpoint:
                self.args.continue_from = latest_checkpoint
        if self.args.continue_from:  # Starting from previous model
            self.state = TrainingState.load_state(state_path=self.args.continue_from, network=self.net_work)
            self.state.init_results_tracking(epochs=self.args.epochs)
            self.model = self.state.model
            if self.args.finetune:
                self.state.init_finetune_states(self.args.epochs)
        else:
            self._build_model()
            self.state = TrainingState(model=self.model)
            self.state.init_results_tracking(epochs=self.args.epochs)

    def _wav2vec_model(self):
        self.vec_model = load_wav2vec_model(self.args.vector_model_path)
        self.vec_model.eval()

    def _init_optimizer_and_device(self):
        self.model = self.model.to(self.device)
        parameters = self.model.parameters()
        if self.args.adam:
            self.optimizer = torch.optim.AdamW(parameters,
                                               lr=self.args.lr,
                                               betas=self.args.betas,
                                               eps=self.args.eps,
                                               weight_decay=self.args.wd)
        else:
            self.optimizer = torch.optim.SGD(parameters,
                                             lr=self.args.lr,
                                             momentum=self.args.momentum,
                                             nesterov=True,
                                             weight_decay=self.args.wd)
        if self.args.continue_from:
            self.recover_learning_rate_from_continue()
        if self.args.cuda and apex_state:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                                                        opt_level=self.args.opt_level,
                                                        loss_scale=self.args.loss_scale)
            self.state.track_optim_state(self.optimizer)
            self.state.track_amp_state(amp)
        #
        if self.args.cuda and self.args.distributed:
            if apex_state:
                self.model = convert_syncbn_model(self.model).to(self.device)
                self.model = DistributedDataParallel(self.model)
            else:
                self.model = DistributedDataParallel(self.model,
                                                     device_ids=[self.args.local_rank],
                                                     output_device=self.args.local_rank,
                                                     find_unused_parameters=True)

    def train_step(self, inputs, targets, input_percentages, target_sizes):
        end_time = time.time()
        inputs = inputs.to(self.device)
        # wav2vec
        if self.args.vector_model_arch == "V1":
            z = self.vec_model.feature_extractor(inputs)
            c = self.vec_model.feature_aggregator(z)
        elif self.args.vector_model_arch == "V2":
            c = self.vec_model(inputs, mask=False, features_only=True)["x"]
            c = c.transpose(1, 2)
        else:
            raise ValueError("wrong type of vector_model_arch:{}".format(self.args.vector_model_arch))
        #
        input_sizes = input_percentages.mul(int(c.size(2))).int()
        #
        out, output_sizes, _ = self.model(c, input_sizes)
        # ctc_loss:
        out = out.transpose(0, 1)  # TxNxH
        out = out.log_softmax(-1)
        float_out = out.float()  # ensure float32 for loss
        loss = self.criterion(float_out,
                              targets.to(self.device).long(),
                              output_sizes.to(self.device),
                              target_sizes.to(self.device))
        loss_value = loss.item()

        # Check to ensure valid loss was calculated
        valid_loss, error = check_loss(loss, loss_value)

        del target_sizes, targets, out, float_out

        if valid_loss:
            self.optimizer.zero_grad()
            # compute gradient
            if self.args.cuda and apex_state:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.args.max_norm)
            else:
                loss.backward()
            self.optimizer.step()
        else:
            print(error)
            print('Skipping grad update')
            loss_value = 0

        self.state.avg_loss += loss_value

        # measure elapsed time
        batch_time_ = time.time() - end_time
        # 清理显存占用
        del loss, input_sizes, inputs
        torch.cuda.empty_cache()

        return loss_value, batch_time_

    def valid_step(self, inputs, targets, input_percentages, target_sizes):
        inputs = inputs.to(self.device)
        if self.args.half:
            inputs = inputs.half()
        # wav2vec
        if self.args.vector_model_arch == "V1":
            z = self.vec_model.feature_extractor(inputs)
            c = self.vec_model.feature_aggregator(z)
        elif self.args.vector_model_arch == "V2":
            c = self.vec_model(inputs, mask=False, features_only=True)["x"]
        else:
            raise ValueError("wrong type of vector_model_arch:{}".format(self.args.vector_model_arch))
        #
        input_sizes = input_percentages.mul(int(c.size(2))).int()
        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size.item()

        out, output_sizes = self.model(inputs, input_sizes)
        decoded_output, _ = self.decoder.decode(out.softmax(-1), output_sizes)
        target_strings = self.decoder.convert_to_strings(split_targets)
        return decoded_output, target_strings

    def train(self, train_loader, valid_loader, main_proc, logger):
        for epoch in range(self.state.epoch, self.args.epochs):
            self.model.train()
            end_time = time.time()
            start_epoch_time = time.time()
            self.state.set_epoch(epoch=epoch)
            train_loader.batch_sampler.set_epoch(epoch=epoch)
            train_loader.batch_sampler.reset_training_step(training_step=self.state.training_step)
            for i, (data) in enumerate(train_loader, start=self.state.training_step):
                self.state.set_training_step(training_step=i)
                inputs, targets, input_percentages, target_sizes = data
                self.data_time.update(time.time() - end_time)
                loss_value, batch_time_ = self.train_step(inputs, targets, input_percentages, target_sizes)
                self.losses.update(loss_value, inputs.size(0))
                self.batch_time.update(batch_time_)
                if main_proc:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                           (epoch + 1), (i + 1), len(train_loader),
                           batch_time=self.batch_time, data_time=self.data_time, loss=self.losses))
            epoch_time = time.time() - start_epoch_time
            if main_proc:
                print('Training Summary Epoch: [{0}]\t'
                      'Time taken (s): {epoch_time:.0f}\t'
                      'Average Loss {loss:.3f}\t'.format(epoch + 1, epoch_time=epoch_time, loss=self.state.avg_loss))

            # evaluate
            wer, cer = self.valid(valid_loader)
            self.state.add_results(epoch=epoch,
                                   loss_result=self.state.avg_loss,
                                   wer_result=wer,
                                   cer_result=cer)
            if main_proc:
                print('Validation Summary Epoch: [{0}]\t'
                      'Average WER {wer:.3f}\t'
                      'Average CER {cer:.3f}\t'.format(epoch + 1, wer=wer, cer=cer))
                logger.info('Validation Summary Epoch: [{0}]\t'
                            'Average WER {wer:.3f}\t'
                            'Average CER {cer:.3f}\t'.format(epoch + 1, wer=wer, cer=cer))
            if main_proc and self.args.visdom:
                self.visdom_logger.update(epoch, self.state.result_state)
            if main_proc and self.args.tensorboard:
                self.tensorboard_logger.update(epoch, self.state.result_state, self.model.named_parameters())

            # anneal lr
            self.learning_rate_decay(epoch, decay_step=5)
            # save model
            self.save_checkpoint(epoch, main_proc, cer, wer)

    def valid(self, valid_loader):
        with torch.no_grad():
            self.model.eval()
            total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
            for i, (data) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                try:
                    inputs, targets, input_percentages, target_sizes = data
                    decoded_output, target_strings = self.valid_step(inputs, targets, input_percentages, target_sizes)
                    for x in range(len(target_strings)):
                        transcript, reference = decoded_output[x][0], target_strings[x][0]
                        transcript = transcript.replace('.', '')
                        reference = reference.replace('.', '')
                        wer_inst = self.decoder.wer(transcript, reference)
                        cer_inst = self.decoder.cer(transcript, reference)
                        total_wer += wer_inst
                        total_cer += cer_inst
                        num_tokens += len(reference.split())
                        num_chars += len(reference.replace(' ', ''))
                except Exception as E:
                    print("Error: {}".format(E))
            wer = float(total_wer) / (num_tokens + 1)
            cer = float(total_cer) / (num_chars + 1)
        return wer * 100, cer * 100

    def test(self, test_loader, best_val_checkpoint, logger=None, half=False):
        torch.set_grad_enabled(False)
        if self.args.net_arch == "base":
            self.model = load_model(FCFintune, self.device, best_val_checkpoint, half)
        else:
            self.model = load_model(TDNNet, self.device, best_val_checkpoint, half)
        wer, cer = self.valid(test_loader)
        print('Best Val model Summary with: {} \t'
              'Best WER {wer:.3f}\t'
              'Best CER {cer:.3f}\t'.format(best_val_checkpoint, wer=wer, cer=cer))
        if logger is not None:
            logger.info('Best Val model Summary with: {}\t'
                        'Best WER {wer:.3f}\t'
                        'Best CER {cer:.3f}\t'.format(best_val_checkpoint + 1, wer=wer, cer=cer))

    def count_parameter(self):
        total_size = self.net_work.get_param_size(self.model)
        return total_size

    def save_checkpoint(self, epoch, main_proc, cer, wer):
        if main_proc and self.args.checkpoint:  # Save epoch checkpoint
            self.checkpoint_handler.save_checkpoint_model(epoch=epoch, state=self.state)

        if main_proc and (self.state.best_wer is None or self.state.best_wer > cer):
            self.checkpoint_handler.save_best_model(epoch=epoch, state=self.state)
            self.state.set_best_wer(cer)
            self.state.reset_avg_loss()
        self.state.reset_training_step()

    def recover_learning_rate_from_continue(self):
        for i in range(int((self.state.epoch + 1) / self.args.decay_epoch)):
            for g in self.optimizer.param_groups:
                g['lr'] = g['lr'] / self.args.learning_anneal
        print('Learning rate recovered from continued model to: {lr:.6f},'
              ' decay epoch step is {decay_epoch:}'.format(lr=g['lr'], decay_epoch=self.args.decay_epoch))

    def learning_rate_decay(self, epoch, decay_step=5):
        if (epoch + 1) % decay_step == 0:
            for g in self.optimizer.param_groups:
                g['lr'] = g['lr'] / self.args.learning_anneal
            print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))

    def init_visualizer(self, visdom_logger=None, tensorboard_logger=None):
        self.visdom_logger = visdom_logger
        self.tensorboard_logger = tensorboard_logger
        if self.args.continue_from:
            if self.args.visdom:  # Add previous scores to visdom graph
                self.visdom_logger.load_previous_values(self.state.epoch, self.state.result_state)
            if self.args.tensorboard:  # Previous scores to tensorboard logs
                self.tensorboard_logger.load_previous_values(self.state.epoch, self.state.result_state)

    @staticmethod
    def load_labels(labels_path):
        with open(labels_path) as label_file:
            labels = [i.strip('\n') for i in label_file.readlines()]
            if ' ' not in labels:
                labels.append(' ')
        return labels
