import torch
import fairseq

cp_path = 'models/V1/wav2vec_large.pt'
model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
model = model[0]
model.eval()

wav_input_16khz = torch.randn(1, 100000)

z = model.feature_extractor(wav_input_16khz)
print("z.size:", z.size())

c = model.feature_aggregator(z)
print("c.size:", c.size())

cpc = model(wav_input_16khz)
print("cpc_logits.size:", cpc["cpc_logits"].size())
