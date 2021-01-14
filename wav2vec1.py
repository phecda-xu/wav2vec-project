import torch
import fairseq

cp_path = 'models/wav2vec_large.pt'
model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
model = model[0]
model.eval()

wav_input_16khz = torch.randn(10, 3000)
z = model.feature_extractor(wav_input_16khz)
c = model.feature_aggregator(z)

# print("z:", z)
print("z.size:", z.size())

# print("c:", c)
print("c.size:", c.size())
