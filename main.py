import torch
import fairseq

cp_path = 'wav2vec_large.pt'
model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
model = model[0]
model.eval()

wav_input_16khz = torch.randn(1,30000)
z = model.feature_extractor(wav_input_16khz)
c = model.feature_aggregator(z)

print("z:",z)
print("c:",c)
