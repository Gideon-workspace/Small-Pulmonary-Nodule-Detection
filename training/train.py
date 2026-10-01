"""
Main training script.

Author: Mashego Gideon Mabeloane
"""
print("PYTHON SCRIPT STARTED", flush=True)
import torch
print("TORCH IMPORTED", flush=True)

from datasets.dataloader import create_dataloaders
print("CREATE_DATALOADER IMPORTED",flush=True)
import pandas as pd
from models.faster_rcnn import get_faster_rcnn
#from models.retinanet import get_retinanet
print("RETINANET IMPORTED", flush=True)
#from models.freeze import setup_freeze_schedule
from optimizer import get_optimizer
from engine import fit

print("ALL IMPORTS DONE,ENTERING THE MAIN", flush=True)

def main():

    # --------------------------------------------------
    # Device
    # --------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print(f"Using device : {device}")
    print("=" * 60)

    # --------------------------------------------------
    # Dataloaders
    # --------------------------------------------------
    print("STEP 1: Creating dataloaders...", flush=True)
    train_loader, valid_loader, test_loader = create_dataloaders(
        train_csv="data/train.csv",
        valid_csv="data/valid.csv",
        test_csv="data/test.csv",
        image_dir="data/images",
    )

    print("STEP 1 DONE: Dataloaders created.", flush=True)


    print("STEP 2: Creating RetinaNet...", flush=True)
    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = get_faster_rcnn(use_lip_stem = True,lip_layers=(2,), post_k=1)

    #model = get_retinanet(use_lip_stem = True,lip_layers=(2,), post_k=1)

    # Lock ResNet backbone, unfreeze LIPModule + FPN + RetinaNet Heads
    #setup_freeze_schedule(model, unfreeze_lip=True, unfreeze_fpn=True)

    print("STEP 2 DONE: RetinaNet created.", flush=True)
    model.to(device)
    print("STEP 3 DONE: Model moved to device.", flush=True)

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------
    print("STEP 4: Creating optimizer...", flush=True)
    optimizer, scheduler = get_optimizer(model)

    # --------------------------------------------------
    # Train
    # --------------------------------------------------
    print("STEP 5: Starting training...", flush=True)

    history = fit(

        model=model,

        train_loader=train_loader,

        valid_loader=valid_loader,

        optimizer=optimizer,

        scheduler=scheduler,

        device=device,

        epochs=50,

    )

    history_df = pd.DataFrame(history)

    history_df.to_csv(

    "Fasterrcnn_LIP2_training_history.csv",

    index=False,

    )

    print("Training history saved.")


if __name__ == "__main__":

    main()
