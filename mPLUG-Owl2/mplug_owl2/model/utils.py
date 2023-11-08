from transformers import AutoConfig


def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if 'mplug_owl2' in config and 'mplug_owl2' not in cfg.model_type:
        assert cfg.model_type == 'mplug_owl2'
        print("You are using newer LLaVA code base, while the checkpoint of v0 is from older code base.")
        print("You must upgrade the checkpoint to the new code base (this can be done automatically).")
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "mplug_owl2")
            cfg.architectures[0] = 'LlavaLlamaForCausalLM'
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)