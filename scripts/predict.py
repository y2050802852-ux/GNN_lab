import torch
from gnn_project.utils import (
    load_config,
    build_parser,
    seed_init,
    get_overrides,
    setup_logger,
    get_device,
    save_result_csv,
)

from gnn_project.engine import Trainer
from gnn_project.data import Dataset
from gnn_project.models.all_models import get_model
from gnn_project.models.all_models import get_model
from datetime import datetime
import argparse
import os


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    cli_overrides = get_overrides(args)


    config = load_config(
        "./configs/config.yaml",
        cli_overrides=cli_overrides
    )
    base_dir = config['log']['out_dir']+'/Predict'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = (
        f"Predict_Result{config['model']}_{config['dataset']}_{timestamp}"
    )
    out_dir = os.path.join(base_dir,run_name)

    logger = setup_logger(out_dir=out_dir,name='train')
    seed_init(config)

    dataset = Dataset(name=config['dataset'],path='./data/',cfg=config)
    data = dataset.data
    in_dim = dataset.in_dim
    n_classes = dataset.n_classes

    model = get_model(
        config['model'],
        # in_dim=data.x.size(1),
        in_dim=in_dim,
        # n_classes=int(data.y.max().item())+1,
        n_classes=n_classes,
        hidden=config['hidden'],
        dropout=config['dropout'],
        task=config['task'],
        model_path=config['load_model_path'],
    )
    device = get_device()

    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'],weight_decay=float(config['weight_decay']))
    if config['task'] in ['graph', 'edge']:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    if config['sched']['enable']:
        sched = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['sched']['step_size'],
            gamma=config['sched']['gamma']
        )
    else:
        sched = None
    trainer = Trainer(model,optimizer,criterion,config['earlystop'], sched=sched, patience=config['patience']
                      , logger=logger, dataset=dataset, epochs=config['epochs'],model_cfg=config,out_dir=out_dir,
                      device=device)

    #loss,val_acc = trainer.fit()
    test_acc = trainer.predict(data)