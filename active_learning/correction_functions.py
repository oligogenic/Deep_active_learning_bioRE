import torch
import os

from accelerate import DistributedType
from accelerate.checkpointing import save_accelerator_state, save_custom_state

def get_state_dict(self, model, unwrap=True):
    # Correction obtained from commit https://github.com/huggingface/accelerate/pull/489/commits/20a96334fd3b6e980f420d5da3d34cac71270061
    self.print('New function for get_state_dict')
    is_zero_3 = False
    if self.distributed_type == DistributedType.DEEPSPEED:
        is_zero_3 = self.deepspeed_config["zero_optimization"]["stage"] == 3

    if is_zero_3:
        if model.zero_gather_16bit_weights_on_model_save():
            state_dict = model._zero3_consolidated_16bit_state_dict()
        else:
            raise ValueError(
                "Cannot get 16bit model weights because `stage3_gather_16bit_weights_on_model_save` in DeepSpeed config is False. "
                "To save the model weights in 16bit, set `stage3_gather_16bit_weights_on_model_save` to True in DeepSpeed config file or "
                "set `zero3_save_16bit_model` to True when using `accelerate config`. "
                "To save the full checkpoint, run `model.save_checkpoint(save_dir)` and use `zero_to_fp32.py` to recover weights."
            )
    else:
        if unwrap:
            model = self.unwrap_model(model)
        state_dict = model.state_dict()
        self.print(state_dict)

    if state_dict is not None:
        for k in state_dict:
            if state_dict[k].dtype == torch.float16:
                state_dict[k] = state_dict[k].float()

    return state_dict


def save_state(self, output_dir: str):
    # Correction obtained from commit https://github.com/huggingface/accelerate/pull/489/commits/20a96334fd3b6e980f420d5da3d34cac71270061
    # Check if folder exists
    self.print('New function for save_state')
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    weights = [self.get_state_dict(m, unwrap=False) for m in self._models]
    save_location = save_accelerator_state(
        output_dir, weights, self._optimizers, self._schedulers, self.state.process_index, self.scaler
    )
    for i, obj in enumerate(self._custom_objects):
        save_custom_state(obj, output_dir, i)
    return save_location