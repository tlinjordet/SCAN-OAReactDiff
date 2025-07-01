import os
import re

#ckpt_path = f"checkpoint/{project}/{wandb_logger.experiment.name}" # Use the actual path from your run
ckpt_path = "checkpoint/TSDiffusion-TS1x-All-Confidence/leftnet-8-4500336f5924/"


best_loss = float('inf')
best_checkpoint_path = None

# Regex to extract the val-totloss from the filename
# Assumes your filename format: ddpm-epoch=XXX-val-totloss=X.XXXX-val-MAE=Y.YYYY-val-Pearson=Z.ZZZZ.ckpt
loss_pattern = re.compile(r'val-totloss=([0-9.]+)')

if os.path.exists(ckpt_path):
    for filename in os.listdir(ckpt_path):
        if filename.endswith(".ckpt"):
            match = loss_pattern.search(filename)
            if match:
                current_loss = float(match.group(1))
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_checkpoint_path = os.path.join(ckpt_path, filename)
else:
    print(f"Checkpoint directory not found: {ckpt_path}")

if best_checkpoint_path:
    print(f"Best checkpoint found: {best_checkpoint_path}")
    print(f"With val-totloss: {best_loss}")
    # Now you can load this checkpoint:
    # model = ConfModule.load_from_checkpoint(best_checkpoint_path, **your_model_init_args)
else:
    print("No checkpoints found or parsed.")
