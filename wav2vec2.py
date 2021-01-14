import torch
from fairseq.models.wav2vec import Wav2VecModel

cp = torch.load('../models/wav2vec_small_960h.pt')
model = Wav2VecModel.build_model(cp['args'], task=None)
model.load_state_dict(cp['model'])

wav_input_16khz = torch.randn(10, 3000)
tensors = torch.from_numpy(wav_input_16khz).unsqueeze(0)

z = model.feature_extractor(tensors)
c = model.feature_aggregator(z)
print('c:', c)
