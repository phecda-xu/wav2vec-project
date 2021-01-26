# coding:utf-8
#
#
#
import torch
import argparse
from tqdm import tqdm
from ASR.data_utils.data_loader import AudioDataLoader, SpectrogramDataset
from ASR.model_utils.model import TDNNet, FCFintune
from ASR.model_utils.model import load_wav2vec_model
from ASR.model_utils.utils import load_model
from ASR.model_utils.decoder import GreedyDecoder

parser = argparse.ArgumentParser(description='TDNN transcription')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for testing')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--half', action="store_true",
                    help='Use half precision. This is recommended when using mixed-precision at training time')
parser.add_argument('--top-paths', default=1, type=int, help='number of beams to return')
parser.add_argument('--beam-width', default=10, type=int, help='Beam width to use')
parser.add_argument('--lm-path', default=None, type=str,
                    help='Path to an (optional) kenlm language model for use with beam search (req\'d with trie)')
parser.add_argument('--alpha', default=0.8, type=float, help='Language model weight')
parser.add_argument('--beta', default=1, type=float, help='Language model word bonus (all words)')
parser.add_argument('--cutoff-top-n', default=40, type=int,
                    help='Cutoff number in pruning, only top cutoff_top_n characters with highest probs in '
                    'vocabulary will be used in beam search, default 40.')
parser.add_argument('--cutoff-prob', default=1.0, type=float,
                    help='Cutoff probability in pruning,default 1.0, no pruning.')
parser.add_argument('--lm-workers', default=1, type=int, help='Number of LM processes to use')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--net-arch', dest='net_arch', type=str, default='base', help="base, super, large, medium or small")
parser.add_argument('--model-path', default='models/*.pth', help='Path to model file created by training')
parser.add_argument('--word-form', dest='word_form', type=str, default='sinogram')
parser.add_argument('--reverberation', dest='reverberation', action='store_true', help='add reverberation to audio.')
parser.add_argument('--reverb-prob', default=0.4, type=float, help='Probability of reverberation added per sample')
parser.add_argument('--test-manifest', metavar='DIR',
                    help='path to validation manifest', default='Work/*/data/manifest.*.test')
parser.add_argument('--verbose', action="store_true", help="print out decoded output and error of each sample")
parser.add_argument('--vector-model-arch', default='V1', type=str, help='path to load wav2vec model')
parser.add_argument('--vector-model-path',
                    default='models/V1/wav2vec_large.pt',
                    type=str,
                    help='path to load wav2vec model')


def evaluate(test_loader, device, model, vec_model, decoder, target_decoder, verbose=False, half=False):
    model.eval()
    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
    for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
        try:
            inputs, targets, input_percentages, target_sizes = data
            inputs = inputs.to(device)
            if half:
                inputs = inputs.half()
            # wav2vec
            if args.vector_model_arch == "V1":
                z = vec_model.feature_extractor(inputs)
                c = vec_model.feature_aggregator(z)
            elif args.vector_model_arch == "V2":
                c = vec_model(inputs, mask=False, features_only=True)["x"]
            else:
                raise ValueError("wrong type of vector_model_arch:{}".format(args.vector_model_arch))
            #
            input_sizes = input_percentages.mul(int(c.size(2))).int()
            # unflatten targets
            split_targets = []
            offset = 0
            for size in target_sizes:
                split_targets.append(targets[offset:offset + size])
                offset += size.item()

            out, output_sizes = model(inputs, input_sizes)

            decoded_output, _ = decoder.decode(out.softmax(-1), output_sizes)
            target_strings = target_decoder.convert_to_strings(split_targets)

            for x in range(len(target_strings)):
                transcript, reference = decoded_output[x][0], target_strings[x][0]
                transcript = transcript.replace('.', '')
                reference = reference.replace('.', '')
                wer_inst = decoder.wer(transcript, reference)
                cer_inst = decoder.cer(transcript, reference)
                total_wer += wer_inst
                total_cer += cer_inst
                num_tokens += len(reference.split())
                num_chars += len(reference.replace(' ', ''))
                if verbose:
                    print("Ref:", reference.lower())
                    print("Hyp:", transcript.lower())
                    print("WER:", float(wer_inst) / len(reference.split()),
                          "CER:", float(cer_inst) / len(reference.replace(' ', '')), "\n")
                    # print("CER:", float(cer_inst) / len(reference.replace(' ', '')), "\n")
        except Exception as E:
            print("Error: {}".format(E))
    wer = float(total_wer) / (num_tokens + 1)
    cer = float(total_cer) / (num_chars + 1)
    return wer * 100, cer * 100


if __name__ == '__main__':
    args = parser.parse_args()
    print("***************************************")
    print("Load file:")
    for param, value in args.__dict__.items():
        print("{} = {}".format(param, value))
    print("***************************************")

    torch.set_grad_enabled(False)
    device = torch.device("cuda" if args.cuda else "cpu")
    if args.net_arch == "base":
        model = load_model(FCFintune, device, args.model_path, args.half)
    else:
        model = load_model(TDNNet, device, args.model_path, args.half)
    # teacher model
    vec_model = load_wav2vec_model(args.vector_model_path)
    vec_model.eval()
    if args.decoder == "beam":
        from ASR.model_utils.decoder import BeamCTCDecoder

        decoder = BeamCTCDecoder(model.labels,
                                 lm_path=args.lm_path,
                                 alpha=args.alpha,
                                 beta=args.beta,
                                 cutoff_top_n=args.cutoff_top_n,
                                 cutoff_prob=args.cutoff_prob,
                                 beam_width=args.beam_width,
                                 num_processes=args.lm_workers)
    elif args.decoder == "greedy":
        decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'), word_form=args.word_form)
    else:
        decoder = None
    target_decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'), word_form=args.word_form)
    print(model.audio_conf)
    test_dataset = SpectrogramDataset(audio_conf=model.audio_conf,
                                      manifest_filepath=args.test_manifest,
                                      labels=model.labels,
                                      word_form=args.word_form,
                                      speed_volume_perturb=args.speed_volume_perturb,
                                      reverberation=args.reverberation)
    test_loader = AudioDataLoader(test_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)
    wer, cer = evaluate(test_loader=test_loader,
                        device=device,
                        model=model,
                        vec_model=vec_model,
                        decoder=decoder,
                        target_decoder=target_decoder,
                        verbose=args.verbose,
                        half=args.half)

    print('Test Summary \t'
          'Average WER {wer:.3f}\t'
          'Average CER {cer:.3f}\t'.format(wer=wer, cer=cer))
