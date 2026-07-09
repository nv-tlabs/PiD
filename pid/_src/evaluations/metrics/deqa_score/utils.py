import torch
from transformers import AutoConfig


def extend_list(data_list, n, min_n):
    if min_n == 0:
        return []
    while len(data_list) < n:
        data_list.extend(data_list[: n - len(data_list)])
    return data_list


def find_prefix(input_ids, prefix):
    """
    input_ids: [B, N1], no start token
    prefix: [N2, ], no start token
    """
    len_prefix = prefix.shape[0]  # N2
    input_ids_unfold = input_ids.unfold(1, len_prefix, 1)
    matches = (input_ids_unfold == prefix).all(dim=2)
    matches_int = matches.type(torch.int64)
    indices = torch.where(
        matches.any(dim=1),
        matches_int.argmax(dim=1),
        torch.tensor(-1, dtype=torch.int64),
    )
    assert (indices >= 0).all(), "Some inputs do not contain prefix"
    return indices


def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if "mplug_owl2" in config and "mplug_owl2" not in cfg.model_type:
        assert cfg.model_type == "mplug_owl2"
        print("You are using newer LLaVA code base, while the checkpoint of v0 is from older code base.")
        print("You must upgrade the checkpoint to the new code base (this can be done automatically).")
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "mplug_owl2")
            cfg.architectures[0] = "LlavaLlamaForCausalLM"
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)
