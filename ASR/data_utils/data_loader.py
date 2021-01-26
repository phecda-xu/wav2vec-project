# coding:utf-8
#
#
import os
import sox
import math
import json
import torch
import numpy as np
import soundfile as sf

from tempfile import NamedTemporaryFile
from .adding_reverb import ReverbAugmentor
from torch.utils.data import Dataset, Sampler, DistributedSampler, DataLoader


def load_audio(path):
    sound, sample_rate = sf.read(path, dtype='int16')
    sound = sound.astype('float32') / 32767  # normalize audio
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average
    return sound


class AudioParser(object):
    def parse_transcript(self, transcript_path):
        """
        :param transcript_path: Path where transcript is stored from the manifest file
        :return: Transcript in training/testing format
        """
        raise NotImplementedError

    def parse_audio(self, audio_path):
        """
        :param audio_path: Path where audio is stored from the manifest file
        :return: Audio in training/testing format
        """
        raise NotImplementedError


class NoiseInjection(object):
    def __init__(self,
                 path=None,
                 sample_rate=16000,
                 noise_levels=(0, 0.5)):
        """
        Adds noise to an input signal with specific SNR. Higher the noise level, the more noise added.
        Modified code from https://github.com/willfrey/audio/blob/master/torchaudio/transforms.py
        """
        if not os.path.exists(path):
            print("Directory doesn't exist: {}".format(path))
            raise IOError
        # self.paths = path is not None and librosa.util.find_files(path)
        with open(path) as f:
            self.paths = f.readlines()
        self.sample_rate = sample_rate
        self.noise_levels = noise_levels

    def inject_noise(self, data):
        noise_info_dic = json.loads(np.random.choice(self.paths))
        noise_path = noise_info_dic['audio_filepath']
        noise_level = np.random.uniform(*self.noise_levels)
        return self.inject_noise_sample(data, noise_path, noise_level)

    def inject_noise_sample(self, data, noise_path, noise_level):
        # noise_len = get_audio_length(noise_path)
        noise_len = sox.file_info.duration(noise_path)
        data_len = len(data) / self.sample_rate
        noise_start = np.random.rand() * (noise_len - data_len)
        noise_end = noise_start + data_len
        noise_dst = audio_with_sox(noise_path, self.sample_rate, noise_start, noise_end)
        if len(data) != len(noise_dst):
            data += 0
        else:
            noise_energy = np.sqrt(noise_dst.dot(noise_dst) / noise_dst.size)
            data_energy = np.sqrt(data.dot(data) / data.size)
            data += noise_level * noise_dst * data_energy / noise_energy
        return data


class SpectrogramParser(AudioParser):
    def __init__(self,
                 audio_conf,
                 speed_volume_perturb=False,
                 reverberation=False):
        """
        Parses audio file into spectrogram with optional normalization and various augmentations
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param normalize(default False):  Apply standard mean and deviation normalization to audio tensor
        :param speed_volume_perturb(default False): Apply random tempo and gain perturbations
        :param spec_augment(default False): Apply simple spectral augmentation to mel spectograms
        """
        super(SpectrogramParser, self).__init__()
        self.sample_rate = audio_conf['sample_rate']
        self.speed_volume_perturb = speed_volume_perturb
        self.reverberation = reverberation
        self.noiseInjector = NoiseInjection(audio_conf['noise_dir'], self.sample_rate,
                                            audio_conf['noise_levels']) if audio_conf.get(
            'noise_dir') is not None else None
        self.noise_prob = audio_conf.get('noise_prob')
        self.reverb_prob = audio_conf.get('reverb_prob')
        self.reverb = ReverbAugmentor(min_distance=3, max_distance=5)

    def parse_audio(self, audio_path):
        # os.system("cp {} {}/wav".format(audio_path, os.getcwd()))
        if self.speed_volume_perturb:
            y = load_randomly_augmented_audio(audio_path, self.sample_rate)
            # sf.write('wav/aaaa_{}'.format(os.path.basename(audio_path)), y, self.sample_rate)
        else:
            y = load_audio(audio_path)
        if self.reverberation:
            add_reverb = np.random.binomial(1, self.reverb_prob)
            if add_reverb:
                y = self.reverb.add_reverb(y)
            # sf.write('wav/bbbb_{}'.format(os.path.basename(audio_path)), y, self.sample_rate)
        if self.noiseInjector:
            add_noise = np.random.binomial(1, self.noise_prob)
            if add_noise:
                y = self.noiseInjector.inject_noise(y)
                # sf.write('wav/cccc_{}'.format(os.path.basename(audio_path)), y, self.sample_rate)
        y = torch.FloatTensor(y)
        return y

    def parse_transcript(self, transcript_path):
        raise NotImplementedError


class SpectrogramDataset(Dataset, SpectrogramParser):
    def __init__(self,
                 audio_conf,
                 manifest_filepath,
                 labels,
                 word_form,
                 speed_volume_perturb=False,
                 reverberation=False,
                 min_durations=0.0,
                 max_durations=60.0):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:

        /path/to/audio.wav,/path/to/audio.txt
        ...

        :param audio_conf: Dictionary containing the sample rate
        :param manifest_filepath: Path to manifest csv as describe above
        :param labels: list containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        :param speed_volume_perturb(default False): Apply random tempo and gain perturbations
        :param spec_augment(default False): Apply simple spectral augmentation to mel spectograms
        """
        self.word_form_fict = {"sinogram": "text", "pinyin": "pinyin", "english": "fully_pinyin"}
        with open(manifest_filepath) as f:
            ids = f.readlines()
        ids = [i for i in ids if min_durations < float(json.loads(i)['duration']) <= max_durations]
        self.ids = sorted(ids, key=lambda x: float(json.loads(x)['duration']), reverse=True)
        self.size = len(ids)
        self.word_form = word_form
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        super(SpectrogramDataset, self).__init__(audio_conf, speed_volume_perturb, reverberation)

    def __getitem__(self, index):
        sample = json.loads(self.ids[index])
        # print("sample: {}".format(sample['duration']))
        audio_path, transcripts = sample['audio_filepath'], sample[self.word_form_fict[self.word_form]]
        raw_data = self.parse_audio(audio_path)
        transcript_id = self.parse_transcript(transcripts)
        return raw_data, transcript_id

    def parse_transcript(self, transcript):
        if self.word_form == 'pinyin':
            transcript_id = [self.label_numerical(x) for x in transcript.split(' ')]
        elif self.word_form == 'sinogram' or self.word_form == 'english':
            transcript_id = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        else:
            raise ValueError('wrong word form: {}'.format(self.word_form))
        return transcript_id

    def __len__(self):
        return self.size

    def label_numerical(self, x):
        if self.labels_map.get(x) is not None:
            return self.labels_map.get(x)
        else:
            return self.labels_map.get('.')


def _collate_fn(batch):
    def func(p):
        return p[0].size(0)

    batch = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)
    longest_sample = max(batch, key=func)[0]
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(0)
    inputs = torch.zeros(minibatch_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(0)
        inputs[x].narrow(0, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
    targets = torch.IntTensor(targets)
    return inputs, targets, input_percentages, target_sizes


class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


class DSRandomSampler(Sampler):
    """
    Implementation of a Random Sampler for sampling the dataset.
    Added to ensure we reset the start index when an epoch is finished.
    This is essential since we support saving/loading state during an epoch.
    """

    def __init__(self, dataset, batch_size=1, start_index=0):
        super().__init__(data_source=dataset)

        self.dataset = dataset
        self.start_index = start_index
        self.batch_size = batch_size
        ids = list(range(len(self.dataset)))
        self.bins = [ids[i:i + self.batch_size] for i in range(0, len(ids), self.batch_size)]

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = (
            torch.randperm(len(self.bins) - self.start_index, generator=g)
                .add(self.start_index)
                .tolist()
        )
        for x in indices:
            batch_ids = self.bins[x]
            np.random.shuffle(batch_ids)
            yield batch_ids

    def __len__(self):
        return len(self.bins) - self.start_index

    def set_epoch(self, epoch):
        self.epoch = epoch

    def reset_training_step(self, training_step):
        self.start_index = training_step


class DSDistributedSampler(DistributedSampler):
    """
    Overrides the DistributedSampler to ensure we reset the start index when an epoch is finished.
    This is essential since we support saving/loading state during an epoch.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, start_index=0, batch_size=1):
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank)
        self.start_index = start_index
        self.batch_size = batch_size
        ids = list(range(len(dataset)))
        self.bins = [ids[i:i + self.batch_size] for i in range(0, len(ids), self.batch_size)]
        self.num_samples = int(
            math.ceil(float(len(self.bins) - self.start_index) / self.num_replicas)
        )
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = (
            torch.randperm(len(self.bins) - self.start_index, generator=g)
                .add(self.start_index)
                .tolist()
        )
        # print("self.bins : {}".format(self.bins))

        indices = sorted(indices, reverse=False)
        # print("indices : {}".format(indices))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank: self.total_size: self.num_replicas]
        assert len(indices) == self.num_samples
        for x in indices:
            batch_ids = self.bins[x]
            np.random.shuffle(batch_ids)
            yield batch_ids

    def __len__(self):
        return self.num_samples

    def reset_training_step(self, training_step):
        self.start_index = training_step
        self.num_samples = int(
            math.ceil(float(len(self.bins) - self.start_index) / self.num_replicas)
        )
        self.total_size = self.num_samples * self.num_replicas


def audio_with_sox(path, sample_rate, start_time, end_time):
    """
    crop and resample the recording with sox and loads it.
    """
    try:
        with NamedTemporaryFile(suffix=".wav") as tar_file:
            tar_filename = tar_file.name
            sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} trim {} ={} >/dev/null 2>&1".format(path, sample_rate,
                                                                                                   tar_filename,
                                                                                                   start_time,
                                                                                                   end_time)
            os.system(sox_params)
            y = load_audio(tar_filename)
    except Exception as E:
        y = load_audio(path)
    return y


def augment_audio_with_sox(path, sample_rate, tempo, gain):
    """
    Changes tempo and gain of the recording with sox and loads it.
    """
    try:
        with NamedTemporaryFile(suffix=".wav") as augmented_file:
            augmented_filename = augmented_file.name
            sox_augment_params = ["tempo", "{:.3f}".format(tempo), "gain", "{:.3f}".format(gain)]
            sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} {} >/dev/null 2>&1".format(path, sample_rate,
                                                                                          augmented_filename,
                                                                                          " ".join(sox_augment_params))
            os.system(sox_params)
            y = load_audio(augmented_filename)
    except Exception as E:
        y = load_audio(path)
    return y


def load_randomly_augmented_audio(path, sample_rate=16000, tempo_range=(0.85, 1.15),
                                  gain_range=(-6, 8)):
    """
    Picks tempo and gain uniformly, applies it to the utterance by using sox utility.
    Returns the augmented utterance.
    """
    low_tempo, high_tempo = tempo_range
    tempo_value = np.random.uniform(low=low_tempo, high=high_tempo)
    low_gain, high_gain = gain_range
    gain_value = np.random.uniform(low=low_gain, high=high_gain)
    audio = augment_audio_with_sox(path=path, sample_rate=sample_rate,
                                   tempo=tempo_value, gain=gain_value)
    return audio
