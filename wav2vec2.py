import torch
import fairseq

model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(['models/wav2vec2_base.pt'])
model = model[0]
model.eval()

wav_input_16khz = torch.randn(1, 100000)

z = model.feature_extractor(wav_input_16khz)
print("z.size:", z.size())

c = model(wav_input_16khz, mask=False, features_only=True)["x"]
print('c size:', c.size())
