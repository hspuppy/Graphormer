"""
Use pretrained or fine-tuned model to do prediction. 
Output a pandas dataframe file with columns: smiles, y, y_pred, metric1, m2, ..., embeddings
"""
import sys
import logging
import os
import sys
from os import path
from pathlib import Path
from typing import OrderedDict

import coloredlogs
import numpy as np
import ogb
import pandas as pd
import scipy
import torch
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from sklearn.metrics import (explained_variance_score, mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score, roc_auc_score)

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from pretrain import load_pretrained_model

coloredlogs.install()
logger = logging.getLogger(__name__)


def predict(args, use_pretrained, checkpoint_path=None, logger=None):
    USE_GPU = args.gpu
    cfg = convert_namespace_to_omegaconf(args)
    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)
    # initialize task
    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)

    # load model, pretrained or checkpoints
    if use_pretrained:
        model_state = load_pretrained_model(cfg.task.pretrained_model_name)
    else:
        model_state = torch.load(checkpoint_path)["model"]
    model.load_state_dict(
        model_state, strict=True, model_cfg=cfg.model
    )
    del model_state

    if USE_GPU:
        model.to(torch.cuda.current_device())
    # load dataset
    split = args.split
    task.load_dataset(split)
    # check_dataset(task.dataset(split))
    batch_iterator = task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=cfg.dataset.max_tokens_valid,
        max_sentences=cfg.dataset.batch_size_valid,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            model.max_positions(),
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_workers=cfg.dataset.num_workers,
        epoch=0,
        data_buffer_size=cfg.dataset.data_buffer_size,
        disable_iterator_cache=False,
    )
    itr = batch_iterator.next_epoch_itr(
        shuffle=False, set_dataset_epoch=False
    )
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple")
    )

    # infer
    y_pred = []
    y_true = []
    embs = None
    with torch.no_grad():
        model.eval()
        for i, sample in enumerate(progress):
            if USE_GPU:
                sample = utils.move_to_cuda(sample)
            y, emb = model(**sample["net_input"], return_emb=True)
            y = y[:, 0, :].reshape(-1)
            if embs is None:
                embs = emb
            else:
                embs = torch.cat([embs, emb], dim=0)
            y_pred.extend(y.detach().cpu())
            y_true.extend(sample["target"].detach().cpu().reshape(-1)[:y.shape[0]])
            # print(i, sample['target'].shape, sample["net_input"]['batched_data']['idx'].shape, len(y_true))
            # if sample['target'].shape[0] != sample["net_input"]['batched_data']['idx'].shape[0]:
            #     import ipdb; ipdb.set_trace()
            if USE_GPU:
                torch.cuda.empty_cache()

    # save predictions, embeddings, metrics
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    # torch.save(embs, 'out_embs.pt')  # save embeddings
    # logger.warning('embedding saved to out_embs.pt')
    metrics = calculate_metrics(y_true, y_pred)
    logger.info(f'Metrics for {split}: {metrics}')
    return pd.DataFrame.from_dict({
        'smiles':[item['net_input.batched_data'].smiles for item in task.dataset(split)], 
        'y_true':y_true, 
        'y_pred':y_pred, 
        'embedding':embs.numpy().tolist()
        })


def calculate_metrics(y_true, y_pred):
    def smape(A, F):
        return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))
    return OrderedDict({
        # 'r2': r2_score(y_true, y_pred),
        'r2': explained_variance_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred),
        'smape': smape(y_true, y_pred),
        'pcc': scipy.stats.pearsonr(y_true, y_pred)
    })


def predict_for_split(args):
        if args.pretrained_model_name != "none":
            return predict(args, True, logger=logger)
        elif hasattr(args, "save_dir"):
            checkpoint_path = Path(args.save_dir) / 'checkpoint_best.pt'
            logger.info(f"evaluating checkpoint file {checkpoint_path}")
            return predict(args, False, checkpoint_path, logger)


def main():
    parser = options.get_training_parser()
    # TODO: Loop split in (train, valid, test) and combine the results
    parser.add_argument("--split", type=str, default='all')
    parser.add_argument("--gpu", default=False, action='store_true')
    args = options.parse_args_and_arch(parser, modify_parser=None)

    if args.split == 'all':
        # when split == all, do predict for (train, valid, test) and combine the results
        dfs = []
        for split in ['train', 'valid', 'test']:
            args.split = split
            dfs.append(predict_for_split(args))
        df = pd.concat(dfs)
    else:
        df = predict_for_split(args)
    logger.warning(f'Total samples: {len(df)} {args.dataset_name} {args.split}')
    metrics = calculate_metrics(df.y_true, df.y_pred)
    logger.warning('Metrics:\n'+'\n'.join(['%s: %s' % (key, metrics[key]) for key in metrics]))
    # saved_filename = f'{args.dataset_name}.pkl'
    # df.to_pickle(saved_filename, protocol=4)
    saved_filename = f'{args.dataset_name}.tsv.gz'
    df.to_csv(saved_filename, sep='\t', index=False)
    logger.warning(f'saved to {saved_filename}')


if __name__ == '__main__':
    main()
