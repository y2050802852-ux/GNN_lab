'''
生成 训练器 Trainer, train/val 函数
'''

import torch
# from torch_geometric import data
from torch_geometric.data import Data, batch
import json,yaml
from gnn_project.utils import load_config, build_parser,setup_logger,seed_init
import time
import os
from torch.utils.tensorboard import SummaryWriter   # 依赖Tensorboard
from ogb.graphproppred import Evaluator

class Trainer():
    def __init__(self,model, opt, criterion, earlystop, sched, patience, logger, dataset, epochs,out_dir,model_cfg,device):
        self.model = model
        self.opt = opt
        self.earlystop = earlystop
        if self.earlystop:
            self.earlystopper = EarlyStopper(patience)
        self.dataset = dataset
        self.epochs = epochs
        self.criterion = criterion
        self.sched = sched
        self.logger = logger
        self.cfg = model_cfg
        self.device = device
        self.start_epoch = 0
        self.best_val = -1
        self.out_dir = out_dir

        self.model = self.model.to(self.device)
        self.dataset.data = self.dataset.data.to(self.device)
        train_out_path = self.out_dir
        os.makedirs(train_out_path,exist_ok=True)

        self.pt_path = os.path.join(train_out_path,f'{self.cfg["model"]}_best.pt')


        #self.pt_path = os.path.join(model_cfg['log']['out_dir'], 'best.pt')
        if self.cfg['log']['tensorboard']:
            self.writer = make_writer(model_cfg,train_out_path)
        else:
            self.writer = None
        self.grad_norm = 0
        seed_init(model_cfg)


    def train_epoch(self,data):
        self.model.train()
        self.opt.zero_grad()
        logits = self.model(data.x, data.edge_index,)
        loss = self.criterion(
            logits[data.train_mask],
            data.y[data.train_mask]
        )
        pred = logits[data.train_mask].argmax(1)
        acc = (pred == data.y[data.train_mask]).float().mean()
        loss.backward()
        self.grad_norm = self.get_model_grad_norm()
        self.opt.step()

        return loss.item(), acc.item()

    def train_sage_epoch(self,loader):
        self.model.train()

        total_loss = 0.
        total_correct = 0.
        total_nodes = 0

        for batch_data in loader:
            batch_data = batch_data.to(self.device)
            self.opt.zero_grad()

            logits = self.model(batch_data.x, batch_data.edge_index)

            n = batch_data.batch_size
            loss = self.criterion(
                logits[:n], batch_data.y[:n]
            )
            loss.backward()
            self.grad_norm = self.get_model_grad_norm()
            self.opt.step()

            total_loss += loss.item() * n
            pred = logits[:n].argmax(1)
            total_correct += (pred == batch_data.y[:n]).sum().item()
            total_nodes += n
        return (total_loss/total_nodes, total_correct/total_nodes)

    def train_graph_epoch(self,loader):
        self.model.train()
        total_loss = 0.
        total_graphs = 0
        
        for batch in loader:
            batch = batch.to(self.device)
            self.opt.zero_grad()

            logits = self.model(
                batch.x,
                batch.edge_index,
                batch.batch,
            )
            y = batch.y.float().view_as(logits)
            mask = torch.isfinite(y)

            if not mask.any():
                continue

            loss = self.criterion(logits[mask], y[mask])
            loss.backward()
            self.grad_norm = self.get_model_grad_norm()
            self.opt.step()

            n = batch.num_graphs
            total_loss += loss.item() * n
            total_graphs += n

        if total_graphs == 0:
            raise RuntimeError("No valid graph labels found in training data.")

        # 返回平均 loss: 训练集 ROC-AUC 可以另行统计
        return total_loss / total_graphs, None

    def train_edge_epoch(self,data):
        self.model.train()
        #data = data.to(self.device)

        self.opt.zero_grad()

        z = self.model(data.x,data.edge_index)
        edge_logits = self.model.decode(z,data.edge_label_index)
        labels = data.edge_label.float().view_as(edge_logits)

        loss = self.criterion(edge_logits,labels)

        loss.backward()
        self.opt.step()

        return loss.item(), None
    
    @torch.no_grad()
    def evaluate_epoch(self,data):
        data = data.to(self.device)
        self.model.eval()

        logits = self.model(data.x,data.edge_index)
        pred = logits[data.val_mask].argmax(1)
        acc = (pred == data.y[data.val_mask]).float().mean()
        return acc.item()

    @torch.no_grad()
    def evaluate_sage_epoch(self,loader):
        self.model.eval()
        total_correct = 0.
        total_nodes = 0
        for batch_data in loader:
            batch_data = batch_data.to(self.device)
            logits = self.model(batch_data.x, batch_data.edge_index)
            n = batch_data.batch_size
            pred = logits[:n].argmax(1)
            total_correct += (pred == batch_data.y[:n]).sum().item()
            total_nodes += n
        return total_correct/total_nodes

    @torch.no_grad()
    def evaluate_graph_epoch(self, loader):
        from ogb.graphproppred import Evaluator

        self.model.eval()
        evaluator = Evaluator(name="ogbg-molhiv")

        y_true_list = []
        y_pred_list = []

        for batch in loader:
            batch = batch.to(self.device)

            logits = self.model(
                batch.x,
                batch.edge_index,
                batch.batch,
            )

            y = batch.y.float().view_as(logits)

            # 转到 CPU，准备交给 OGB Evaluator
            y_true_list.append(y.cpu())
            y_pred_list.append(logits.sigmoid().cpu())

        y_true = torch.cat(y_true_list, dim=0).numpy()
        y_pred = torch.cat(y_pred_list, dim=0).numpy()

        return evaluator.eval({
            "y_true": y_true,
            "y_pred": y_pred,
        })["rocauc"]

    @torch.no_grad()
    def evaluate_edge_epoch(self,data):
        data = data.to(self.device)
        self.model.eval()
        z = self.model(data.x,data.edge_index)
        edge_logits = self.model.decode(z,data.edge_label_index)
        pred = (edge_logits > 0 ).long()
        correct = (pred == data.edge_label.long()).sum().item()
        return correct / data.edge_label.numel()
    
    def save(self,epoch, val_acc):
        '''保存模型参数'''
        torch.save({
            'epoch':epoch, 'model':self.model.state_dict(),
            'opt':self.opt.state_dict(), 'sched': self.sched.state_dict() if self.sched is not None else None,
            'cfg':self.cfg, 'val_acc':val_acc, 
            'time':time.strftime('%F %T'),
            'earlystopper': {
            'best_val': self.earlystopper.best_val,
            'bad_epochs': self.earlystopper.bad_epochs,
            'stopped': self.earlystopper.stopped,
            } if self.earlystop else None
        }, self.pt_path)
        self.logger.info(f" save to {self.out_dir} @ epoch {epoch}, val {val_acc}")

    def resume(self, path):
        '''恢复模型参数'''
        ck = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ck['model'])
        self.opt.load_state_dict(ck['opt'])
        if ck.get('sched') is not None and self.sched is not None:
            self.sched.load_state_dict(ck['sched'])
        self.start_epoch = ck['epoch'] + 1
        self.best_val = ck['val_acc']

        early_state = ck.get('earlystopper')
        if self.earlystop and early_state is not None:
            self.earlystopper.best_val = self.best_val
            self.earlystopper.bad_epochs = early_state['bad_epochs']
            self.earlystopper.stopped = early_state['stopped']

        self.logger.info(f"resumed from {path} @ epoch {self.start_epoch}, best_val {self.best_val:.4f}")

    def train_step(self, task):
        if self.cfg['model'].lower() == 'sage' and task == 'node':
            loss, acc = self.train_sage_epoch(self.dataset.train_loader)
        elif task.lower() == 'graph':
            loss, acc = self.train_graph_epoch(self.dataset.train_loader)
        elif task.lower() == 'edge':
            loss, acc = self.train_edge_epoch(self.dataset.data)
        else:
            loss, acc = self.train_epoch(self.dataset.data)
        return loss, acc

    def test(self,task):
        if self.cfg['model'].lower() == 'sage' and task == 'node':
            test_acc = self.evaluate_sage_epoch(self.dataset.test_loader)
        elif task.lower() == 'graph':
            test_acc = self.evaluate_graph_epoch(self.dataset.test_loader)
        elif task.lower() == 'edge':
            test_acc = self.evaluate_edge_epoch(self.dataset.test_data)
        else:
            test_acc = self.evaluate_epoch(self.dataset.test_data)
        return test_acc


    def fit(self,resume_need=False, resume_path=None):
        self.logger.info(f"Training is starting: Model: {self.cfg['model']}, Dataset: {self.cfg['dataset']}, Device: {self.device}")
        if resume_need:
            if resume_path:
                self.resume(path=resume_path)
            else:
                self.resume(path=self.pt_path)
        loss_history = []
        val_acc_history =[]
        try:
            for epoch in range(self.start_epoch,self.epochs):

                #loss, acc = self.train_epoch(self.dataset.data)
                loss, acc = self.train_step(self.cfg['task'])
                self.logger.info(f"TRAIN Result: epoch: {epoch}: loss:{loss}, train_acc:{acc}")
                #val_acc = self.evaluate_epoch(data)

                val_acc = self.evaluate_step(self.cfg['task'])
                if epoch % 5 == 0:
                    
                    self.logger.info(f"VAL Result: epoch: {epoch}: val_acc:{val_acc}")
                if self.sched is not None:
                    if isinstance(self.sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.sched.step(val_acc)
                    else:
                        self.sched.step()
                cur_lr = self.opt.param_groups[0]['lr']
                trian_write(self.writer,epoch,loss,val_acc,cur_lr,self.grad_norm)

                if val_acc > self.best_val:
                    self.best_val = val_acc
                    self.save(epoch, val_acc)

                if self.earlystop:
                    _, is_stop = self.earlystopper.step(val_acc)
                else:
                    is_stop = False
                if is_stop:
                    self.logger.info(f"Triggered Early Stopping")
                    break

                loss_history.append(loss)
                val_acc_history.append(val_acc)
        except KeyboardInterrupt as E :
            self.logger.warning(f"Triggered {E}, Auto-save Complete!!!")
            self.save(epoch,val_acc)
        finally:   
            self.writer.close()
        self.logger.info(f"Training Result saved on {self.out_dir}")
        return loss_history, val_acc_history

    def get_model_grad_norm(self):
        grads = [
        p.grad.detach()
        for p in self.model.parameters()
        if p.grad is not None
        ]

        grad_norm = (
            torch.stack([g.norm(2) for g in grads]).norm(2).item()
            if grads else 0.0
        )
        return grad_norm


    def evaluate_step(self,task):
        if self.cfg['model'].lower() == 'sage' and task == 'node':
            val_acc = self.evaluate_sage_epoch(self.dataset.valid_loader)
        elif task.lower() == 'graph':
            val_acc = self.evaluate_graph_epoch(self.dataset.valid_loader)
        elif task.lower() == 'edge':
            val_acc = self.evaluate_edge_epoch(self.dataset.valid_data)
        else:
            val_acc = self.evaluate_epoch(self.dataset.valid_data)
        return val_acc

    @torch.no_grad()
    def predict(self, data):
        self.model.eval()

        task = self.cfg['task'].lower()
        data = data.to(self.device)

        if task == 'node':
            pred = self.model(data.x,data.edge_index)

        elif task == 'graph':
            pred = self.model(data)
        elif task == 'edge':
            z = self.model(data.x,data.edge_index)
            pred = self.model.decode(z,data.edge_label_index)
        else:
            raise ValueError(f"Unsupported task: {task}")

        out_dir = self.out_dir+'/Predict'
        os.makedirs(self.out_dir+'/Predict',exist_ok=True)
        save_path = os.path.join(out_dir,'predictions.pt')

        if isinstance(pred,torch.Tensor):
            predict_to_save = pred.detach().cpu()
        else:
            predict_to_save = torch.as_tensor(pred).cpu()
        torch.save(predict_to_save,save_path)
        self.logger.info(f"Model prediction result: {pred}!!!\n result saved to {save_path}")
        return pred
            


class EarlyStopper:
    def __init__(self,patience=30, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta

        self.best_val = float('-inf')
        self.bad_epochs = 0
        self.stopped = False


    def step(self, val_metric):
        # 本轮是否真的改善？
        improved = val_metric > self.best_val + self.min_delta

        if improved:
            self.best_val = val_metric
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            self.stopped = True

        return improved, self.stopped    



def make_writer(cfg,out_dir):
    run_name = f"{cfg['model']}-{cfg['hidden']}-lr{cfg['lr']}-seed{cfg['seed']}"
    log_dir = os.path.join(out_dir)
    return SummaryWriter(log_dir+'/TensorBoard',run_name)



def trian_write(writer,epoch, train_loss,val_acc,cur_lr,grad_norm):
    if writer is not None:
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('acc/val',  val_acc, epoch)
        writer.add_scalar('lr', cur_lr, epoch)
        writer.add_scalar('grad/norm', grad_norm, epoch)
    else:
        pass