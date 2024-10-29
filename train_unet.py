import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import urllib.request
from torch.utils.tensorboard import SummaryWriter
import datetime
import torchvision.transforms.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np

# Add the project directory to the Python path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_dir not in sys.path:
    sys.path.append(project_dir)

from Unet_implementation.data_loader import Arcade  
from Unet_implementation.unet_model import UNet  

class Augment:
    def __call__(self, image, mask):
        if torch.rand(1) > 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)
        if torch.rand(1) > 0.5:
            image = F.vflip(image)
            mask = F.vflip(mask)
        if torch.rand(1) > 0.5:
            angle = torch.randint(-30, 30, (1,)).item()
            image = F.rotate(image, angle)
            mask = F.rotate(mask, angle)
        return image, mask

def dice(pred, true, threshold=0.5):
    pred = (pred > threshold).astype(np.float32)
    intersection = np.sum(pred[true == 1]) * 2.0
    dice_score = intersection / (np.sum(pred) + np.sum(true))
    return dice_score

def combined_loss(preds, targets, bce_weight=0.5):
    bce = nn.BCEWithLogitsLoss()(preds, targets)
    preds_np = torch.sigmoid(preds).detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    dice_val = dice(preds_np, targets_np)
    return bce * bce_weight + (1 - dice_val) * (1 - bce_weight)

def download_weights(url, destination):
    if not os.path.exists(destination):
        print(f"Downloading pretrained weights from {url}...")
        urllib.request.urlretrieve(url, destination)
        print("Download completed.")
    else:
        print("Pretrained weights already exist.")

class ArcadeModel(pl.LightningModule):
    def __init__(self, pretrained_path=None, learning_rate=0.001, bce_weight=0.5):
        super(ArcadeModel, self).__init__()
        self.model = UNet(encoder_name='resnet50', pretrained=True, pretrained_path=pretrained_path)
        self.learning_rate = learning_rate
        self.bce_weight = bce_weight

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = combined_loss(outputs, masks, self.bce_weight)

        preds = torch.sigmoid(outputs).detach() > 0.5
        preds_np = preds.cpu().numpy()
        masks_np = masks.detach().cpu().numpy()

        dice_val = dice(preds_np, masks_np)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_dice', dice_val, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx == 0:
            self.log_images(images, masks, preds, 'train')

        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = combined_loss(outputs, masks, self.bce_weight)

        preds = torch.sigmoid(outputs).detach() > 0.5
        preds_np = preds.cpu().numpy()
        masks_np = masks.detach().cpu().numpy()

        dice_val = dice(preds_np, masks_np)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_dice', dice_val, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx == 0:
            self.log_images(images, masks, preds, 'val')

        return {'val_loss': loss, 'val_dice': dice_val}

    def test_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = combined_loss(outputs, masks, self.bce_weight)

        preds = torch.sigmoid(outputs).detach() > 0.5
        preds_np = preds.cpu().numpy()
        masks_np = masks.detach().cpu().numpy()

        dice_val = dice(preds_np, masks_np)

        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_dice', dice_val, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx == 0:
            self.log_images(images, masks, preds, 'test')

        return {'test_loss': loss, 'test_dice': dice_val}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.2)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def log_images(self, images, masks, preds, stage):
        grid_image = torchvision.utils.make_grid(images)
        grid_mask = torchvision.utils.make_grid(masks)
        grid_pred = torchvision.utils.make_grid(preds.float())

        self.logger.experiment.add_image(f'{stage}_images', grid_image, self.current_epoch)
        self.logger.experiment.add_image(f'{stage}_masks', grid_mask, self.current_epoch)
        self.logger.experiment.add_image(f'{stage}_preds', grid_pred, self.current_epoch)

def visualize_sample(image, mask, prediction):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image.permute(1, 2, 0).cpu())
    axs[0].set_title('Image')
    axs[1].imshow(mask.squeeze().cpu(), cmap='gray')
    axs[1].set_title('Ground Truth Mask')
    axs[2].imshow(prediction.squeeze().cpu(), cmap='gray')
    axs[2].set_title('Predicted Mask')
    plt.show()

def main(image_dir, mask_dir, pretrained_path=None, num_epochs=80, batch_size=4, learning_rate=0.001):
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory '{image_dir}' not found")
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Mask directory '{mask_dir}' not found")

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    augmentation = Augment()

    train_dataset = Arcade(image_dir, mask_dir, transform=transform, augmentation=augmentation)
    val_dataset = Arcade(image_dir_val, mask_dir_val, transform=transform, augmentation=augmentation)
    test_dataset = Arcade(image_dir_test, mask_dir_test, transform=transform, augmentation=augmentation)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = ArcadeModel(pretrained_path=pretrained_path, learning_rate=learning_rate)
    
    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = TensorBoardLogger("tb_logs", name=log_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",  # Directory to save the best model
        filename='best_model',  # Filename of the best model
        monitor='val_dice',     # Metric to monitor (here it’s validation dice score)
        mode='max',             # Save the model with the highest 'val_dice'
        save_top_k=1,           # Save only the best model (top 1)
        verbose=True            # Display information when the model is saved
    )

    trainer = pl.Trainer(
        max_epochs=num_epochs, 
        devices=1 if torch.cuda.is_available() else None, 
        accelerator='gpu' if torch.cuda.is_available() else 'cpu', 
        logger=logger, 
        callbacks=[checkpoint_callback]
    )
    
    trainer.fit(model, train_loader, val_loader)

    # Load the best checkpoint
    best_model_path = checkpoint_callback.best_model_path
    model = ArcadeModel.load_from_checkpoint(best_model_path)

    # Test the model
    trainer.test(model, dataloaders=test_loader)
    
    print(f'Finished Training. Best model saved at {best_model_path}')

if __name__ == '__main__':
    image_dir = "/home/vault/iwi5/iwi5208h/my_thesis/segmentation/arcade/syntax/train/images"
    mask_dir = "/home/vault/iwi5/iwi5208h/my_thesis/segmentation/arcade/syntax/train/masks"
    image_dir_val = "/home/vault/iwi5/iwi5208h/my_thesis/segmentation/arcade/syntax/val/images"
    mask_dir_val = "/home/vault/iwi5/iwi5208h/my_thesis/segmentation/arcade/syntax/val/masks"
    image_dir_test = "/home/vault/iwi5/iwi5208h/my_thesis/segmentation/arcade/syntax/test/images"
    mask_dir_test = "/home/vault/iwi5/iwi5208h/my_thesis/segmentation/arcade/syntax/test/masks"
    
    main(image_dir, mask_dir)
