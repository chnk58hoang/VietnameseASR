import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf

cfg = OmegaConf.load('configs/conformers/conformer_ctc_bpe_stream.yaml')
ssl_model = nemo_asr.models.SpeechEncDecSelfSupervisedModel.from_pretrained(model_name="ssl_en_conformer_large")

# define down-stream model
asr_model = nemo_asr.models.EncDecCTCModelBPE(cfg=cfg.model)

# load ssl checkpoint
asr_model.load_state_dict(ssl_model.state_dict(), strict=False)

# discard ssl model
del ssl_model