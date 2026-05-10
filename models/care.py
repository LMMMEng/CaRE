import copy
import logging
import warnings
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.distributions.multivariate_normal import MultivariateNormal


from models.base import BaseLearner
from utils.inc_net import CaRENet
from utils.toolkit import target2onehot, tensor2numpy


num_workers = 8

class AngularPenaltySMLoss(nn.Module):
    def __init__(self, loss_type='cosface', eps=1e-7, s=20, m=0):
        super().__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface', 'crossentropy']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 20.0 if not s else s
            self.m = 0.0 if not m else m

        self.loss_type = loss_type
        self.eps = eps

    def forward(self, wf, labels):
        if self.loss_type == 'crossentropy':
            return F.cross_entropy(wf * self.s, labels)
        else:
            if self.loss_type == 'cosface':
                numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
            if self.loss_type == 'arcface':
                numerator = self.s * torch.cos(torch.acos(
                    torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)) + self.m)
            if self.loss_type == 'sphereface':
                numerator = self.s * torch.cos(self.m * torch.acos(
                    torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)))

            excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
            denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
            L = numerator - torch.log(denominator)
            return -torch.mean(L)


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

        self._network = CaRENet(args, True)
        self.args = args
        self.cls_mean = dict()
        self.cls_cov = dict()
        self.cls2task = dict()

        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args.get("weight_decay", 5e-4)
        self.min_lr = args.get("min_lr", 1e-8)
        self.tuned_epoch = args['tuned_epoch']
        # self.args['tuned_epoch'] = args['tuned_epoch']
        self.ca_lr = args["ca_lr"]
        self.crct_epochs = args["crct_epochs"]
        self.self_dist_loss = args.get("self_dist_loss", True)
        self.aux_cos_loss = args.get("aux_cos_loss", True)
        
        # AMP training
        try:
            self.use_amp = args["amp"]
        except:
            self.use_amp = False
            logging.info("Do not find AMP setting in config, set to False as default")

        self.scaler = GradScaler(enabled=self.use_amp)
        
        if self.use_amp:
            logging.info("Training with Automatic Mixed Precision")

        try:
            self.iter_pred = args['iter_pred']
        except:
            self.iter_pred = True

        for n, p in self._network.backbone.named_parameters():
            if 'adapter' not in n and 'head' not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True

        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        logging.info(f'{total_params:,} model total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} model training parameters.')

        # if some parameters are trainable, print the key name and corresponding parameter number
        if total_params != total_trainable_params:
            for name, param in self._network.named_parameters():
                if param.requires_grad:
                    logging.info("{}: {}".format(name, param.numel()))

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):

        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        for i in range(self._known_classes, self._total_classes):
            self.cls2task[i] = self._cur_task

        self._network.update_fc(self._total_classes - self._known_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                      source="train", mode="train")
        self.data_manager = data_manager
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)


        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        
        min_lr = None

        if self._cur_task > 0:
            num_classes = self._total_classes - self._known_classes
            self._network.backbone.update_adapters(num_classes)
            self._network.backbone.train_shared_adapter(use_ema=True)
            num_epochs = self.tuned_epoch // 2
            min_lr = self.init_lr / 2
        else:
            num_epochs = self.tuned_epoch
            self._network.backbone.train_shared_adapter(use_ema=False)
            
        self._network.backbone.to(self._device)
        self._network.fc.to(self._device)

        optimizer = self.get_optimizer(self._network.backbone)
        
        scheduler = self.get_scheduler(optimizer, min_lr=min_lr, num_epochs=num_epochs)
        self._init_train(train_loader, test_loader, optimizer, scheduler, num_epochs)

        if self._cur_task > 0:

            self._compute_mean(self._network.backbone)
            self.classifer_align()
            lr = optimizer.param_groups[0]['lr']
            optimizer = self.get_optimizer(self._network.backbone, lr=lr)
            scheduler = self.get_scheduler(optimizer, num_epochs=num_epochs)
            self._network.backbone.train_shared_adapter(use_ema=True)
            self._init_train(train_loader, test_loader, optimizer, scheduler, num_epochs)

        self._compute_mean(self._network.backbone)

        if self._cur_task > 0:
            self.classifer_align()


    def get_optimizer(self, model, lr=None):
        if lr is None:
            lr = self.init_lr
        base_params = [p for name, p in model.named_parameters() if 'adapter' in name and p.requires_grad]
        base_params = {'params': base_params, 
                       'lr': lr, 
                       'weight_decay': self.weight_decay}
        base_fc_params = {'params': self._network.fc.parameters(), 
                          'lr': lr,
                          'weight_decay': self.weight_decay}
        network_params = [base_params, base_fc_params]

        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                network_params,
                momentum=0.9,
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                network_params,
            )

        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                network_params,
            )

        return optimizer

    def get_scheduler(self, optimizer, min_lr=None, num_epochs=None):
        
        if min_lr is None:
            min_lr = self.min_lr
        
        if num_epochs is None:
            num_epochs = self.tuned_epoch

        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, 
                                                             T_max=num_epochs,
                                                             eta_min=min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, 
                                                       milestones=self.args["init_milestones"],
                                                       gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def _init_train(self, train_loader, test_loader, optimizer, scheduler, num_epochs):
        # prog_bar = tqdm(range(self.args['tuned_epoch']))
        prog_bar = tqdm(range(num_epochs))
        loss_cos = AngularPenaltySMLoss(loss_type='cosface', eps=1e-7, s=self.args["scale"], m=self.args["m"])

        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()
            losses = 0.0
            losses_main = 0.0
            losses_aux = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):

                inputs, targets = inputs.to(self._device), targets.to(self._device)
                
                with autocast(enabled=self.use_amp):

                    output = self._network.backbone(inputs, adapter_id=self._cur_task, with_task_id=True)
                    features = output["features"]
                    aux_logits = output["latents"]
                    
                    logits = self._network.fc(features)["logits"]
                    loss_main = loss_cos(logits[:, self._known_classes:], targets - self._known_classes)

                    loss_aux = []
                    teacher = logits[:, self._known_classes:].clone().detach()
                    
                    for idx, item in enumerate(aux_logits):
                        
                        if self.aux_cos_loss:
                            hard_loss_aux = loss_cos(item, targets - self._known_classes)
                        else:
                            hard_loss_aux = F.cross_entropy(item, targets - self._known_classes)
                        
                        _loss_aux = hard_loss_aux

                        if self.self_dist_loss:

                            t = 1 if idx < 6 else 2
                            t_probs = torch.softmax(teacher / t, dim=1)
                            s_probs = torch.log_softmax(item / t, dim=1)
                            soft_loss_aux = F.kl_div(s_probs, t_probs.detach(), reduction='batchmean') * (t ** 2)
                            _loss_aux = (_loss_aux + soft_loss_aux) / 2
  
                        loss_aux.append(_loss_aux)

                    loss_aux = sum(loss_aux) / len(loss_aux)
                    loss = loss_main + loss_aux

                optimizer.zero_grad()
                
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                
                losses += loss.item()
                losses_main += loss_main.item()
                losses_aux += loss_aux.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            lr = optimizer.param_groups[0]['lr']
            
            info = f'Task {self._cur_task}, Epoch {epoch + 1}/{num_epochs} => LR {lr:.2e} ' + \
                   f'Loss {losses / len(train_loader):.3f}/{losses_main / len(train_loader):.3f}/{losses_aux / len(train_loader):.3f}, ' + \
                   f'Train_accy {train_acc:.2f}'

            prog_bar.set_description(info)

        logging.info(info)
    
    
    @torch.no_grad()
    def _compute_mean(self, model):
        model.eval()
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = self.data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self.batch_size * 3, shuffle=False, num_workers=4
            )

            vectors = []
            for _, _inputs, _targets in idx_loader:
                with autocast(enabled=False):
                    _vectors = model(_inputs.to(self._device), adapter_id=self._cur_task, with_task_id=True)["features"]
                vectors.append(_vectors)
            vectors = torch.cat(vectors, dim=0)

            if self.args["ca_storage_efficient_method"] == 'covariance':
                features_per_cls = vectors
                self.cls_mean[class_idx] = features_per_cls.mean(dim=0).to(self._device)
                self.cls_cov[class_idx] = torch.cov(features_per_cls.T) + (
                        torch.eye(self.cls_mean[class_idx].shape[-1]) * 1e-4).to(self._device)
            elif self.args["ca_storage_efficient_method"] == 'variance':
                features_per_cls = vectors
                self.cls_mean[class_idx] = features_per_cls.mean(dim=0).to(self._device)
                self.cls_cov[class_idx] = torch.diag(
                    torch.cov(features_per_cls.T) + (torch.eye(self.cls_mean[class_idx].shape[-1]) * 1e-4).to(
                        self._device))


    def classifer_align(self):
        for p in self._network.fc.parameters():
            p.requires_grad = True

        run_epochs = self.crct_epochs
        network_params = [
            {'params': self._network.fc.parameters(), 'lr': self.ca_lr, 'weight_decay': self.weight_decay}]
        optimizer = optim.SGD(network_params, lr=self.ca_lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=5)

        prog_bar = tqdm(range(run_epochs))
        task_size = self._known_classes - self._total_classes
        self._network.eval()

        for epoch in prog_bar:

            sampled_data = []
            sampled_label = []
            num_sampled_pcls = self.batch_size * 5

            if self.args["ca_storage_efficient_method"] in ['covariance', 'variance']:
                for class_idx in range(self._total_classes):
                    if self.args["decay"]:
                        t_id = class_idx // task_size
                        decay = (t_id + 1) / (self._cur_task + 1) * 0.1
                        mean = self.cls_mean[class_idx].detach().clone().to(torch.float64) * (0.9 + decay)
                    else:
                        mean = self.cls_mean[class_idx].to(self._device)
                    cov = self.cls_cov[class_idx].to(self._device)
                    if self.args["ca_storage_efficient_method"] == 'variance':
                        cov = torch.diag(cov)
                    m = MultivariateNormal(mean.float(), cov.float())
                    sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                    sampled_data.append(sampled_data_single)
                    sampled_label.extend([class_idx] * num_sampled_pcls)

            else:
                raise NotImplementedError

            sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
            sampled_label = torch.tensor(sampled_label).long().to(self._device)
            if epoch == 0:
                print("sampled data shape: ", sampled_data.shape)

            inputs = sampled_data
            targets = sampled_label

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            losses = 0.0
            correct, total = 0, 0
            for _iter in range(self._total_classes):

                inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                
                with autocast(enabled=False):
                    outputs = self._network.fc(inp)["logits"]
                    logits = self.args['scale'] * outputs
                    loss = F.cross_entropy(logits, tgt)

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(tgt.expand_as(preds)).cpu().sum()
                total += len(tgt)

                optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                
                losses += loss

            scheduler.step()
            ca_acc = np.round(tensor2numpy(correct) * 100 / total, decimals=2)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, CA_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.crct_epochs,
                losses / self._total_classes,
                ca_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)
    
    
    @torch.no_grad()
    def _eval_cnn_singe_pass(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            with autocast(enabled=False):
                features = self._network.backbone(inputs, adapter_id=0, with_task_id=False)["features"]
                logits = self._network.fc(features)["logits"][:, :self._total_classes]

            predicts = torch.topk(logits, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return (np.concatenate(y_pred), np.concatenate(y_true))  # [N, topk]
    

    @torch.no_grad()
    def _eval_cnn(self, loader):
        self._network.eval()
        max_iter_pred_task_id = 60
        '''
        Iterative inference is affordable when the task sequence is still short.
        Once the task id grows beyond this point, its evaluation cost scales with the
        number of learned task adapters, so we switch to the single-pass path for
        better long-sequence efficiency. In practice, we explicitly disable this branch in 
        long sequences (omnibench-1k) to keep evaluation runtime under control.
        '''
        if (not self.iter_pred) or (self._cur_task == 0) or (self._cur_task > max_iter_pred_task_id):
            y_pred, y_true = self._eval_cnn_singe_pass(loader)
            return (y_pred, y_true)
        else:
            y_pred, y_true = [], []

            for _, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                all_predicts = []
                all_entropies = []
                all_logits = []

                for i in range(self._cur_task + 1):
                    
                    with autocast(enabled=False):
                        features = self._network.backbone(inputs, adapter_id=i, with_task_id=True)["features"]
                        logits = self._network.fc(features)["logits"][:, :self._total_classes]*self.args['scale']

                    probs = torch.softmax(logits, dim=1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)  # bs
                    predicts = torch.topk(logits, k=self.topk, dim=1, largest=True, sorted=True)[1]

                    all_predicts.append(predicts)
                    all_entropies.append(entropy)
                    all_logits.append(logits)
                    
                all_entropies = torch.stack(all_entropies, dim=0)
                all_logits = torch.stack(all_logits, dim=0)
                
                min_entropy_indices = torch.argmin(all_entropies, axis=0)  # bs
                min_entropy_logits = all_logits[min_entropy_indices, torch.arange(len(min_entropy_indices))].to(self._device)
                min_entropy_logits = torch.softmax(min_entropy_logits, dim=1)

                with autocast(enabled=False):
                    features = self._network.backbone(inputs, adapter_id=None, with_task_id=False)["features"]
                    logits = self._network.fc(features)["logits"][:, :self._total_classes]*self.args['scale']
                
                logits = torch.softmax(logits, dim=1)

                outputs = logits + min_entropy_logits

                predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]
                y_pred.append(predicts.cpu().numpy())
                y_true.append(targets.cpu().numpy())

            return (np.concatenate(y_pred), np.concatenate(y_true))