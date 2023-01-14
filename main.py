import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from data import WiderFaceDetection, preproc
from layers.functions.prior_box import PriorBox
from layers.modules import MultiBoxLoss
from models.retinaface import RetinaFace


class RetinaFaceModule(pl.LightningModule):
    def __init__(self
        image_size,
    ):
        super().__init__()
        self.image_size = image_size

    def _build_model(self):
        self.model = RetinaFace(cfg=cfg)
        priorbox = PriorBox(cfg, image_size=(self.image_size, self.image_size))
        with torch.no_grad():
            self.priors = priorbox.forward()

    def _build_loss(self):
        self.criterion = MultiBoxLoss(2, 0.35, True, 0, True, 7, 0.35, False)
    
    def _build_metrics(self):
        ...

    def compute_loss(self, preds, gts):
        loss_l, loss_c, loss_landm = self.criterion(preds, self.priors, gts)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        return loss

    def training_step(self, batch, batch_idx):
        images, gts = batch
        preds = self.model(images)
        loss = self.compute_loss(preds, gts)
        return loss

    def test_step(self, batch, batch_idx)


if __name__ == "__main__":
    dataset = WiderFaceDetection( training_dataset,preproc(img_dim, rgb_mean))
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate)
