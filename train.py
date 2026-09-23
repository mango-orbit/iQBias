import argparse
import json
import os
import time
import warnings
import yaml
import contextlib
from pathlib import Path

import numpy as np
import termcolor
from tqdm import tqdm
import options as option_lib
import utilities.misc as misc

### ---------------------------------------------------------------
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import timm
import architectures as archs
import datasampler as dsamplers
import datasets as dataset_library
import criteria as criteria
import metrics as metrics
import batchminer as bmine
from utilities import misc
from utilities.metric_plotter import TrainingMetricPlotter
from utilities.result_run_guard import ActiveRunGuard
from utilities.experiment_change_logger import (
    ExperimentChangeLogger, mark_active_failed, mark_active_interrupted)
from datasets.basic_dataset import BaseDataset

from metrics import query_gallery_metrics

warnings.filterwarnings("ignore")

# define datasets with query and gallery
QUERY_GALLERY_DATASETS = ['inshop']
QUERY_GALLERY_RECALL_KS = [1, 10, 20, 30, 40, 50]

# define datasets need memory bank
LARGE_DATASETS = ['inshop', 'sop']


def fast_evaluation_recall_ks(dataset_name):
    """Return the standard fast-evaluation Recall cutoffs for a dataset."""
    normalized_name = dataset_name.lower()
    if normalized_name == 'sop' or 'online_products' in normalized_name:
        return [1, 10, 100, 1000]
    return [1, 2, 4, 8]


def create_optimizer(opt, to_optim, optimizer_name=None):
    optimizer_name = optimizer_name or opt.optim
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(to_optim)
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(to_optim)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(to_optim,
                                momentum=opt.sdg_momentum)
    else:
        raise Exception(
            'Optimizer <{}> not available!'.format(optimizer_name))
    return optimizer

def create_scheduler(opt, scheduler, optimizer):
    if scheduler == 'multi':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=opt.lr_reduce_multi_steps,
                                                 gamma=opt.lr_reduce_rate)
    elif scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=opt.lr_reduce_step,
                                            gamma=opt.lr_reduce_rate)
    elif scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                            start_factor=opt.lr_linear_start,
                                            end_factor=opt.lr_linear_end,
                                            total_iters=opt.lr_linear_length)
    elif scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                            T_max=opt.lr_cosine_length,
                                            eta_min=opt.lr_cosine_min)
    else:
        raise Exception('Scheduler <{}> not available!'.format(opt.scheduler))
    return scheduler


def _gradient_pair_stats(first, second):
    """Return norms and cosine for two same-length gradient lists."""
    device = next(
        (gradient.device for gradient in first + second
         if gradient is not None), torch.device('cpu'))
    first_sq = torch.zeros((), device=device, dtype=torch.float64)
    second_sq = torch.zeros((), device=device, dtype=torch.float64)
    dot = torch.zeros((), device=device, dtype=torch.float64)
    for first_gradient, second_gradient in zip(first, second):
        if first_gradient is not None:
            first_float = first_gradient.detach().double()
            first_sq = first_sq + first_float.square().sum()
        else:
            first_float = None
        if second_gradient is not None:
            second_float = second_gradient.detach().double()
            second_sq = second_sq + second_float.square().sum()
        else:
            second_float = None
        if first_float is not None and second_float is not None:
            dot = dot + (first_float * second_float).sum()
    first_norm = first_sq.sqrt()
    second_norm = second_sq.sqrt()
    denominator = first_norm * second_norm
    cosine = (dot / denominator).item() if denominator.item() > 0 else float('nan')
    return first_norm.item(), second_norm.item(), cosine


def log_nir_gradient_diagnostics(model, criterion, epoch):
    """Inspect loss-branch gradients without modifying accumulated .grad."""
    proxy_loss = getattr(criterion, 'diagnostic_proxy_loss', None)
    nir_loss = getattr(criterion, 'diagnostic_weighted_nir_loss', None)
    if proxy_loss is None or nir_loss is None:
        return

    groups = []
    head_parameters = []
    for attribute in ('head', 'fc_norm', 'dist_head', 'dist_norm'):
        module = getattr(model, attribute, None)
        if module is not None:
            head_parameters.extend(
                parameter for parameter in module.parameters()
                if parameter.requires_grad)
    groups.append(('Head', head_parameters))

    proxy_parameter = getattr(model, 'proxies_bias', None)
    proxy_parameters = (
        [proxy_parameter]
        if proxy_parameter is not None and proxy_parameter.requires_grad
        else [])
    groups.append(('Proxy', proxy_parameters))

    flow_parameters = [
        parameter for parameter in criterion.parameters()
        if parameter.requires_grad]
    groups.append(('Flow', flow_parameters))

    all_parameters = [
        parameter for _, parameters in groups for parameter in parameters]
    if not all_parameters:
        return
    proxy_gradients = torch.autograd.grad(
        proxy_loss, all_parameters, retain_graph=True, allow_unused=True)
    nir_gradients = torch.autograd.grad(
        nir_loss, all_parameters, retain_graph=True, allow_unused=True)

    offset = 0
    output = []
    for name, parameters in groups:
        length = len(parameters)
        proxy_slice = list(proxy_gradients[offset:offset + length])
        nir_slice = list(nir_gradients[offset:offset + length])
        offset += length
        proxy_norm, nir_norm, cosine = _gradient_pair_stats(
            proxy_slice, nir_slice)
        cosine_text = 'nan' if np.isnan(cosine) else '{:+.4f}'.format(cosine)
        ratio = nir_norm / proxy_norm if proxy_norm > 0 else float('nan')
        ratio_text = 'nan' if np.isnan(ratio) else '{:.4f}'.format(ratio)
        output.append(
            '{} PA={:.3e} NIR={:.3e} Ratio={} Cos={}'.format(
                name, proxy_norm, nir_norm, ratio_text, cosine_text))
    print('[NIR Gradient Ep:{:02d}] {}'.format(epoch, ' | '.join(output)))

    # Drop the extra references; the combined loss still owns the graph used
    # by the normal scaled backward immediately after this diagnostic.
    criterion.diagnostic_proxy_loss = None
    criterion.diagnostic_weighted_nir_loss = None



def train():
    
    ### INPUT ARGUMENTS
    parser = argparse.ArgumentParser()
    parser = option_lib.basic_training_parameters(parser)
    parser = option_lib.scale_optimizing_parameters(parser)
    parser = option_lib.batch_creation_parameters(parser)
    parser = option_lib.batchmining_specific_parameters(parser)
    parser = option_lib.loss_specific_parameters(parser)
    parser = option_lib.nir_parameters(parser)
    parser = option_lib.simdino_parameters(parser)
    parser = option_lib.ratefit_parameters(parser)
    parser = option_lib.prrd_parameters(parser)
    
    ### Additional, non-default parameters.
    opt = parser.parse_args()
    
    # update setting from config
    with open(opt.config) as file:
        if file is not None:
            _config = yaml.load(file, Loader=yaml.FullLoader)
            for key, value in _config.items():
                setattr(opt, key, value['value'])
    assert opt.world_size == 1, 'Distributed training not yet supported for turn.py!'
    
    # setup auto memory bank
    if opt.eval_interval <= 0:
        raise ValueError('eval_interval must be greater than 0.')
    if opt.eval_before_training not in (0, 1):
        raise ValueError('eval_before_training must be 0 or 1.')
    if opt.fast_evaluation not in (0, 1):
        raise ValueError('fast_evaluation must be 0 or 1.')
    if opt.disable_plots not in (0, 1):
        raise ValueError('disable_plots must be 0 or 1.')
    if opt.progress_update_interval <= 0:
        raise ValueError('progress_update_interval must be greater than 0.')
    # Checkpoint/result bookkeeping used to assume Euclidean Recall@1.  Hyp
    # experiments expose h_recall@1 instead, so use the configured primary
    # storage metric throughout the run.
    primary_metric_name = (opt.storage_metrics[0]
                           if opt.storage_metrics else 'e_recall@1')
    if opt.subset_experiment_enabled not in (0, 1):
        raise ValueError('subset_experiment_enabled must be 0 or 1.')
    if opt.subset_experiment_enabled:
        if opt.quick_train_class_count <= 0:
            raise ValueError(
                'quick_train_class_count must be greater than 0.')
        if opt.quick_test_class_count <= 0:
            raise ValueError(
                'quick_test_class_count must be greater than 0.')
        print(
            'Quick experiment enabled: protocol={} | first {} official '
            'training classes | first {} official test classes | normal '
            'evaluation interval={} epoch(s)'.format(
                opt.subset_experiment_protocol or 'unnamed',
                opt.quick_train_class_count, opt.quick_test_class_count,
                opt.eval_interval))
    if opt.fast_evaluation:
        # Not only fewer plots: the expensive clustering and mAP metrics are
        # removed at the MetricComputer entry point.
        recall_ks = fast_evaluation_recall_ks(opt.dataset)
        recall_prefix = ('h_recall' if getattr(opt, 'hyp_metric', False)
                         or opt.loss == 'hyp_pairwise' else 'e_recall')
        opt.evaluation_metrics = [
            '{}@{}'.format(recall_prefix, recall_k) for recall_k in recall_ks]
        if str(getattr(opt, 'beft_stage_a_output', '') or '').strip():
            # Keep the template Recall evaluator and add the Stage-A MAP@R
            # record without enabling plots or training-set evaluation.
            opt.evaluation_metrics.append('mAP_R')
        print(
            'Fast evaluation enabled: {} + best records only.'.format(
                '/'.join('Recall@{}'.format(k) for k in recall_ks)))
    plots_enabled = not bool(opt.disable_plots)
    auxiliary_plots_enabled = plots_enabled and not bool(opt.fast_evaluation)
    if not plots_enabled:
        print('Plotting disabled: metric records remain enabled.')
    if opt.heatmap_max_images <= 0:
        raise ValueError("heatmap_max_images must be greater than 0.")
    if opt.model_input_size <= 0:
        raise ValueError("model_input_size must be greater than 0.")
    if opt.amp_dtype not in ('float16', 'bfloat16'):
        raise ValueError("amp_dtype must be float16 or bfloat16.")
    if opt.warmup < 0:
        raise ValueError("warmup must be a non-negative integer.")
    if opt.loss in ('nir', 'nir_dml_proxy_peft',
                    'nir_dml_proxy_peft_v2'):
        # ``loss_nir_w_align`` is retained for old configs.  New experiments
        # use the explicit PA name so it is clear which half of
        # L = L_flow + w_pa * L_PA is being reweighted.
        if opt.loss_nir_pa_weight is None:
            opt.loss_nir_pa_weight = opt.loss_nir_w_align
        if opt.loss_nir_pa_weight < 0:
            raise ValueError("loss_nir_pa_weight must be non-negative.")
        if opt.loss_nir_proxy_lrmulti <= 0:
            raise ValueError("loss_nir_proxy_lrmulti must be greater than 0.")
        if opt.loss_nir_lrmulti <= 0:
            raise ValueError("loss_nir_lrmulti must be greater than 0.")
        if opt.loss_nir_nf_cblocks <= 0:
            raise ValueError("loss_nir_nf_cblocks must be greater than 0.")
        if opt.loss_nir_nf_fc_depth < 2:
            raise ValueError("loss_nir_nf_fc_depth must be at least 2.")
        if opt.loss_nir_nf_fc_width <= 0:
            raise ValueError("loss_nir_nf_fc_width must be greater than 0.")
        if opt.loss_nir_dequantization_noise < 0:
            raise ValueError(
                "loss_nir_dequantization_noise must be non-negative.")
    if opt.loss == 'nir_dml_proxy_peft_v2':
        if opt.loss_nir_aux_weight < 0:
            raise ValueError("loss_nir_aux_weight must be non-negative.")
        if opt.loss_nir_feature_ramp_epochs < 0:
            raise ValueError(
                "loss_nir_feature_ramp_epochs must be non-negative.")
        if opt.loss_nir_relative_negatives < 1:
            raise ValueError(
                "loss_nir_relative_negatives must be at least 1.")
        if opt.loss_nir_relative_margin < 0:
            raise ValueError(
                "loss_nir_relative_margin must be non-negative.")
        if opt.loss_nir_relative_temperature <= 0:
            raise ValueError(
                "loss_nir_relative_temperature must be greater than 0.")
        if opt.loss_nir_gradient_diagnostics not in (0, 1):
            raise ValueError(
                "loss_nir_gradient_diagnostics must be 0 or 1.")
    bias_tuning_mode = str(getattr(opt, 'bias_mask', 'auto')).lower()
    standard_bitfit_active = (
        bool(opt.all_bias_bitfit) if bias_tuning_mode == 'auto'
        else bias_tuning_mode == 'bitfit')
    bitfit_modes = 1 if standard_bitfit_active else 0
    if opt.bitfit_scope != 'all' and not standard_bitfit_active:
        raise ValueError(
            'A non-default bitfit_scope currently requires --all_bias_bitfit.')
    stage_a_audit_enabled = bool(
        str(getattr(opt, 'beft_stage_a_output', '') or '').strip())




    if opt.batch_size >= 256 and opt.dataset in LARGE_DATASETS:
        opt.prompt_memory_bank = True
        opt.checkpoint = True
    
    if opt.batch_size >= 256 and opt.embed_dim >= 512:
        opt.prompt_memory_bank = True
        opt.checkpoint = True
    
    metric_plotter = TrainingMetricPlotter(
        opt.save_path, opt.dataset, opt.pretrained_model,
        heatmap_max_images=opt.heatmap_max_images,
        tsne_num_classes=opt.tsne_num_classes,
        tsne_samples_per_class=opt.tsne_samples_per_class,
        auxiliary_plots=not opt.fast_evaluation,
        plots_enabled=plots_enabled,
        evaluation_metrics=opt.evaluation_metrics)
    # Mark this run directory precisely. The result cleaner validates the PID
    # and the process start clock, so a long-running directory is never
    # mistaken for an interrupted one.
    run_guard = ActiveRunGuard(metric_plotter.run_dir, opt.name)
    archived_config_path = metric_plotter.archive_run_config(opt.config)
    opt.config = str(archived_config_path)
    print("Run configuration: {}".format(archived_config_path))
    effective_parameters_path = (
        metric_plotter.run_dir / "effective_parameters.yaml")
    temporary_parameters_path = (
        metric_plotter.run_dir / ".effective_parameters.yaml.tmp")
    with open(temporary_parameters_path, "w", encoding="utf-8") as output_file:
        yaml.safe_dump(
            dict(sorted(vars(opt).items())), output_file,
            allow_unicode=True, sort_keys=False)
    os.replace(str(temporary_parameters_path), str(effective_parameters_path))
    print("Effective parameters: {}".format(effective_parameters_path))
    if plots_enabled:
        print('Metric curve directory: {}'.format(metric_plotter.image_dir))

    experiment_logger = None
    try:
        experiment_logger = ExperimentChangeLogger(
            metric_plotter.run_dir, effective_parameters_path)
        print("Experiment change log: {}".format(
            experiment_logger.log_path))
        print("Dataset daily log directory: {}".format(
            experiment_logger.logs_dir))
    except Exception as error:
        # Logging is auxiliary: a logging failure must never stop training.
        print("Warning: failed to initialize experiment log: {}".format(
            error))

    def refresh_experiment_log():
        if experiment_logger is None:
            return
        try:
            experiment_logger.update_metrics()
        except Exception as error:
            print("Warning: failed to refresh experiment log: {}".format(
                error))

    opt.source_path += '/' + opt.dataset
    opt.save_path += '/' + opt.dataset

    # Assert that the construction of the batch makes sense, i.e. the division into
    # class-subclusters.
    assert_text = 'Batchsize needs to fit number of samples per class for distance '
    assert_text += 'sampling and margin/triplet loss!'
    assert not opt.batch_size % opt.samples_per_class, assert_text
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu[0])
    misc.set_seed(opt.seed)
    misc.configure_cudnn(opt.cudnn_benchmark)
    if opt.cudnn_benchmark:
        print('cuDNN benchmark enabled: autotuning fixed input shapes; '
              'deterministic cuDNN convolution selection disabled.')
    else:
        print('cuDNN legacy mode: benchmark disabled; deterministic cuDNN '
              'convolution selection disabled.')

    ### ---------------------------------------------------------------
    ### Embedding Network model (resnet, inception).
    
    # timm backbones take their preprocessing from pretrained_cfg; the copied
    # configuration is handed to the dataset loader.
    opt.pretrained_cfg = None
    if opt.arch == 'nir_resnet50':
        opt.pretrained_cfg = {
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
            'input_size': (3, opt.model_input_size, opt.model_input_size),
        }
    elif opt.arch != 'nir_bninception':
        # The input resolution is passed explicitly so that timm resamples the
        # absolute position embeddings and the patch grid accordingly.
        _model_for_cfg = timm.create_model(
            opt.pretrained_model, pretrained=False,
            img_size=opt.model_input_size)
        if hasattr(_model_for_cfg, 'pretrained_cfg'):
            opt.pretrained_cfg = _model_for_cfg.pretrained_cfg
        del _model_for_cfg
    
    ####### Datasetse & Dataloaders.
    
    datasets = dataset_library.select(opt)
    opt.n_classes = datasets['training'].n_classes

    tsne_dataset = None
    if auxiliary_plots_enabled:
        tsne_image_dict = {}
        for image_path, class_id in datasets['training'].image_list:
            tsne_image_dict.setdefault(class_id, []).append(image_path)
        tsne_dataset = BaseDataset(tsne_image_dict, opt, is_validation=True)
    
    opt.device = torch.device('cuda')
    model = archs.select(opt.arch, opt)
    
    if hasattr(model, 'optim_dict_list') and len(model.optim_dict_list):
        to_optim = model.optim_dict_list
    else:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        to_optim = [{'params': model_parameters, 'lr': opt.lr, 'weight_decay': opt.decay}]
    
    ####### Build Criterion.
    batchminer = bmine.select(opt.batch_mining, opt)
    criterion, to_optim = criteria.select(opt.loss, opt, to_optim, batchminer)
    _ = criterion.to(opt.device)
    
    if opt.show_trainable_pars:
        model._print_trainable_params()
    
    _ = model.to(opt.device)

    dataloaders = {}
    if not opt.dataset in QUERY_GALLERY_DATASETS:
        dataloaders['evaluation'] = torch.utils.data.DataLoader(
                datasets['evaluation'],
                num_workers=opt.kernels,
                batch_size=opt.batch_size,
                pin_memory=True,
                shuffle=False)
        dataloaders['testing'] = torch.utils.data.DataLoader(
                datasets['testing'],
                num_workers=opt.kernels,
                batch_size=opt.batch_size,
                pin_memory=True,
                shuffle=False)
    else:
        dataloaders['query'] = torch.utils.data.DataLoader(
                datasets['query'],
                num_workers=opt.kernels,
                batch_size=opt.batch_size,
                pin_memory=True,
                shuffle=False)
        dataloaders['gallery'] = torch.utils.data.DataLoader(
                datasets['gallery'],
                num_workers=opt.kernels,
                batch_size=opt.batch_size,
                pin_memory=True,
                shuffle=False)
    
    train_data_sampler = dsamplers.select(opt.data_sampler, opt,
                                      datasets['training'].image_dict,
                                      datasets['training'].image_list)
    
    if train_data_sampler.requires_storage:
        train_data_sampler.create_storage(dataloaders['evaluation'], model,
                                      opt.device)
    
    dataloaders['training'] = torch.utils.data.DataLoader(
        datasets['training'],
        num_workers=opt.kernels,
        pin_memory=True,
        batch_sampler=train_data_sampler)
    if auxiliary_plots_enabled:
        dataloaders['tsne'] = torch.utils.data.DataLoader(
            tsne_dataset,
            num_workers=opt.kernels,
            batch_size=opt.batch_size,
            pin_memory=True,
            shuffle=False)
    
    if not opt.dataset in QUERY_GALLERY_DATASETS:
        opt.n_test_classes = len(dataloaders['testing'].dataset.avail_classes)
    
        metric_evaluation_keys = ['testing', 'evaluation']
    
    
    ### ---------------------------------------------------------------
    if opt.prompt_memory_bank:
        model.p_prompts.to('cpu')
    
    ### Optimizer ####
    model.optimizer = create_optimizer(opt, to_optim)
    model.p_optimizer = create_optimizer(opt, model.p_optim_dict_list)
    criterion_optimizer = None
    if hasattr(criterion, 'separate_optim_dict_list'):
        criterion_optimizer = create_optimizer(
            opt, criterion.separate_optim_dict_list,
            optimizer_name=criterion.separate_optimizer_name)
        print('NIR Flow optimizer: {} (independent from PEFT {})'.format(
            criterion.separate_optimizer_name, opt.optim))
    scheduler = create_scheduler(opt, opt.scheduler, model.optimizer)

    if getattr(model, 'nir_standalone', False):
        # A model that owns a standalone auxiliary objective uses the configured
        # schedule for both optimisers; otherwise the proxy parameters keep the
        # fixed cosine schedule used here.
        p_scheduler = create_scheduler(opt, opt.scheduler, model.p_optimizer)
    else:
        p_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            model.p_optimizer, T_max=opt.lr_cosine_length,
            eta_min=opt.lr_cosine_min)
    
    ### Metric Computer.
    metric_computer = metrics.MetricComputer(opt.evaluation_metrics, opt)
    
    ### ---------------------------------------------------------------
    ### Summary.
    trainable_par = misc.gimme_params(model)
    extra_par = misc.gimme_params(criterion)
    summary = 'Dataset:\t {} \n'.format(opt.dataset)
    summary += 'Objective:\t {} \n'.format(opt.loss)
    summary += 'Backbone:\t {} \n'.format(opt.arch)
    summary += 'Pretrained:\t {} \n'.format(opt.pretrained_model)
    summary += 'Optimizer:\t {} \n'.format(opt.optim)
    summary += 'Trainable Pars:\t {:.2f}M ({}) \n'.format(trainable_par/1e6, trainable_par)
    summary += 'Extra Pars:\t {:.2f}M ({}) \n'.format(extra_par/1e6, extra_par)
    if hasattr(model, 'bitfit_effective_bias_count') and bitfit_modes:
        summary += 'Optimizer Backbone Bias Pars:\t {} \n'.format(
            model.bitfit_optimizer_bias_count)
        summary += 'Effective Selected Bias Pars:\t {} \n'.format(
            model.bitfit_effective_bias_count)
        summary += 'Eligible Non-Key Bias Pars:\t {} \n'.format(
            model.bitfit_eligible_bias_count)
    if getattr(model, 'bias_mask', '') in (
            'q', 'k', 'v', 'v_out',
            'qv', 'v_fc2'):
        bias_counts = model.bias_parameter_accounting()
        summary += 'BEFT Optimizer-visible Trainable Pars:\t {} \n'.format(
            bias_counts['optimizer_visible_trainable_params'])
        summary += 'BEFT Effective Trainable Pars:\t {} \n'.format(
            bias_counts['effective_trainable_params'])
        summary += 'BEFT Effective Backbone Pars:\t {} \n'.format(
            bias_counts['backbone_effective_trainable_params'])
    print(summary)
    ### ----------------------------------------------------------------
    
    #### Main training. #####
    print('\n' + termcolor.colored('===============================================', 'red', 
        attrs=['bold']) + '\n')
    
    iter_count = 0
    loss_args = {
        'batch': None,
        'batch_view2': None,
        'labels': None,
        'nir_proxies': None,
        'batch_features': None,
        'f_embed': None,
        'rc_features': None,
        'backbone_features': None,
        'reference_backbone_features': None,
        'head_relation_embeddings': None,
        'reference_head_embeddings': None,
    }

    opt.epoch = 0
    epochs = range(opt.epoch, opt.n_epochs)
    
    amp_dtype = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }[opt.amp_dtype]
    if (opt.mix_precision and amp_dtype == torch.bfloat16
            and hasattr(torch.cuda, 'is_bf16_supported')
            and not torch.cuda.is_bf16_supported()):
        raise RuntimeError(
            'BF16 is not supported by this GPU; set amp_dtype=float16 or disable '
            'mix_precision.')
    # FP16 benefits from gradient scaling. BF16 already has FP32-like range.
    scaler = torch.cuda.amp.GradScaler(
        enabled=opt.mix_precision and amp_dtype == torch.float16)
    context = (torch.cuda.amp.autocast(dtype=amp_dtype)
               if opt.mix_precision else contextlib.nullcontext())
    print('Mixed precision: {}{}'.format(
        'enabled, dtype=' if opt.mix_precision else 'disabled',
        opt.amp_dtype if opt.mix_precision else ''))
    
    torch.cuda.empty_cache()

    stage_a_audit = None
    if stage_a_audit_enabled:
        from utilities.bias_audit import StageABiasAudit
        stage_a_audit = StageABiasAudit(
            model, opt, Path(opt.beft_stage_a_output))



    def save_best_peft_state(epoch, recall_at_one):
        """Atomically save only task-specific/trainable parameters."""
        # Stage audits require a recoverable best-epoch artifact regardless
        # of the legacy BitFit-only checkpoint flag.  This changes logging
        # only; it does not alter optimization or model selection metrics.
        if not (opt.bitfit_save_peft_checkpoint or
                stage_a_audit_enabled):
            return
        state = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        payload = {
            'format_version': 1,
            'epoch': int(epoch),
            'recall_at_one': float(recall_at_one),
            'experiment_name': opt.name,
            'pretrained_model': opt.pretrained_model,
            'selection_manifest': (
                model.bitfit_selection_manifest
                if hasattr(model, 'bitfit_selection_manifest') else None),
            'trainable_state_dict': state,
        }
        checkpoint_path = metric_plotter.records_dir / 'best_peft_state.pt'
        temporary_path = checkpoint_path.with_name(
            '.best_peft_state.tmp.pt')
        torch.save(payload, temporary_path)
        os.replace(str(temporary_path), str(checkpoint_path))

    def compute_evaluation_metrics(include_training_metrics=False, label="Testing"):
        """Evaluate the current model and return metrics plus ordered test features."""
        _ = model.eval()
        print("\n" + termcolor.colored(
            "Computing {} Metrics...".format(label),
            "green", attrs=["bold"]))

        heatmap_features = None
        heatmap_labels = None
        if opt.dataset not in QUERY_GALLERY_DATASETS:
            with context:
                computed_metrics, test_extra_infos = metric_computer.compute_standard(
                    opt, model, dataloaders["testing"], opt.evaltypes,
                    opt.device, mode="Val",
                    return_features=auxiliary_plots_enabled)

            if auxiliary_plots_enabled:
                for evaltype in opt.evaltypes:
                    if evaltype in test_extra_infos:
                        test_info = test_extra_infos[evaltype]
                        heatmap_features = test_info.get("features")
                        heatmap_labels = test_info.get("target_labels")
                        if heatmap_features is not None:
                            break

            numeric_metrics = {}
            for main_key in computed_metrics.keys():
                for name, value in computed_metrics[main_key].items():
                    if main_key not in numeric_metrics:
                        numeric_metrics[main_key] = {}
                    numeric_metrics[main_key][name] = value

            if include_training_metrics:
                computed_metrics, _ = metric_computer.compute_standard(
                    opt, model, dataloaders["evaluation"], opt.evaltypes,
                    opt.device, mode="Train", return_features=False)
                if primary_metric_name in computed_metrics["embeds"]:
                    numeric_metrics["embeds"]["train_recall@1"] = \
                        computed_metrics["embeds"][primary_metric_name]
                if "mAP_R" in computed_metrics["embeds"]:
                    numeric_metrics["embeds"]["train_mAP_R"] = \
                        computed_metrics["embeds"]["mAP_R"]
        else:
            with context:
                query_gallery_result = \
                    query_gallery_metrics.evaluate_query_gallery(
                        model, dataloaders["query"], dataloaders["gallery"],
                        QUERY_GALLERY_RECALL_KS,
                        return_features=auxiliary_plots_enabled)
            if auxiliary_plots_enabled:
                (recalls, map_value, heatmap_features,
                 heatmap_labels) = query_gallery_result
            else:
                recalls, map_value = query_gallery_result
            numeric_metrics = {"embed": {}}
            for k, recall in zip(QUERY_GALLERY_RECALL_KS, recalls):
                numeric_metrics["embed"]["e_recall@{}".format(k)] = recall
            numeric_metrics["embed"]["MAP"] = map_value

        return numeric_metrics, heatmap_features, heatmap_labels

    def compute_tsne_inputs():
        """Extract deterministic training samples and their current class proxies."""
        _ = model.eval()
        selected_classes = dataloaders["tsne"].dataset.avail_classes[
            :opt.tsne_num_classes]
        collected_features = {class_id: [] for class_id in selected_classes}

        with torch.no_grad():
            for inp in dataloaders["tsne"]:
                input_images, targets = inp[1]["image"], inp[0]
                with context:
                    batch_features = model(input_images.to(
                        opt.device, non_blocking=True))["embeds"]
                batch_features = batch_features.detach().cpu()
                targets = targets.detach().cpu()
                for class_id in selected_classes:
                    remaining = (opt.tsne_samples_per_class -
                                 len(collected_features[class_id]))
                    if remaining <= 0:
                        continue
                    indices = torch.nonzero(
                        targets == class_id, as_tuple=False).flatten()[:remaining]
                    collected_features[class_id].extend(
                        batch_features[indices].unbind(dim=0))
                if all(len(values) >= opt.tsne_samples_per_class
                       for values in collected_features.values()):
                    break

        feature_parts = []
        label_parts = []
        for class_id in selected_classes:
            values = collected_features[class_id]
            if not values:
                continue
            feature_parts.append(torch.stack(values, dim=0))
            label_parts.append(torch.full(
                (len(values),), class_id, dtype=torch.long))
        if not feature_parts:
            raise RuntimeError("No training samples were available for t-SNE.")

        proxy_model = model.module if hasattr(model, "module") else model
        if not hasattr(proxy_model, "proxies_bias"):
            raise RuntimeError("The selected model does not expose class proxies.")
        proxies = proxy_model.proxies_bias
        if opt.semantic_proxy and hasattr(proxy_model, "proxies_container"):
            # Matches the training path in qv_vit._proxies_generator: plot the
            # fused semantic proxies Q_c, not only the learnable proxy bias O_c.
            proxies = (proxies * opt.proxy_bias_ratio +
                       proxy_model.proxies_container *
                       (1.0 - opt.proxy_bias_ratio))
        proxies = torch.nn.functional.normalize(proxies, dim=-1).detach().cpu()
        return (torch.cat(feature_parts, dim=0),
                torch.cat(label_parts, dim=0), proxies)

    def format_evaluation_metrics(numeric_metrics):
        result = ""
        for evaltype, values in numeric_metrics.items():
            result += "Embed-Type: {}:\n".format(evaltype)
            for index, (metric_name, metric_value) in enumerate(values.items()):
                result += "{}{}: {:4.4f}".format(
                    "\n" if index > 0 else "", metric_name, metric_value)
            result += "\n"
        return result

    baseline_metrics = {}
    if opt.eval_before_training:
        # Evaluate the initialized/pretrained model before the first batch.
        baseline_start_time = time.time()
        (baseline_metrics, baseline_features,
         baseline_labels) = compute_evaluation_metrics(
            include_training_metrics=False, label="Epoch 0 Baseline")
        if not auxiliary_plots_enabled:
            baseline_tsne_features = baseline_tsne_labels = baseline_proxies = None
        else:
            baseline_tsne_features, baseline_tsne_labels, baseline_proxies = \
                compute_tsne_inputs()
        metric_plotter.update(
            0, baseline_metrics, baseline_features, baseline_labels,
            baseline_tsne_features, baseline_tsne_labels, baseline_proxies)
        refresh_experiment_log()
        print(format_evaluation_metrics(baseline_metrics))
        print("\nEpoch 0 Evaluation Runtime: {0:4.2f}s".format(
            time.time() - baseline_start_time))
        print("\n" + termcolor.colored(
            "===============================================",
            "red", attrs=["bold"]) + "\n")



    ################# Training Loop #################

    max_r1 = max(
        (values[primary_metric_name] for values in baseline_metrics.values()
         if primary_metric_name in values),
        default=0.0)
    max_map = max(
        (value for values in baseline_metrics.values()
         for name, value in values.items() if name in ("mAP_R", "MAP")),
        default=0.0)
    best_checkpoint_r1 = max_r1
    
    for epoch in epochs:
        opt.epoch = epoch
        warmup_active = epoch < opt.warmup
        if warmup_active:
            print('-- NIR-style warmup: epoch {}/{}; '
                  'freeze sample feature extractor, train DML head/proxies --'.format(
                      epoch + 1, opt.warmup))
        
        misc.set_seed(opt.seed + epoch)
        
        epoch_start_time = time.time()
        epoch_learning_rate = float(model.optimizer.param_groups[0]['lr'])
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        
        # Train one epoch
        data_iterator = tqdm(dataloaders['training'],
                bar_format='{desc}{percentage:2.2f}% [{n_fmt}/{total_fmt}, '
                           '{elapsed}/{remaining}{postfix}]',
                ncols=96, ascii=True, mininterval=1.0,
                desc='[GPU:%d Train Ep:%d]: ' % (opt.gpu[0], epoch))
        
        _ = model.train()
        criterion.new_epoch = True
        
        total_time = 0.0
        avg_time = 0.0
        # for each minibatch
        for i, out in enumerate(data_iterator):

            if opt.time:
                torch.cuda.synchronize()
                start = time.time()
            
            class_labels, input_dict, sample_indices = out

            # load data to generator
            input = input_dict['image']
            
            model_args = {
                'x': input.to(opt.device, non_blocking=True),
                'labels': class_labels.to(opt.device, non_blocking=True),
                'anchor': None,
                'warmup': warmup_active
            }
            
            model.optimizer.zero_grad(set_to_none=True)
            model.p_optimizer.zero_grad(set_to_none=True)
            if criterion_optimizer is not None:
                criterion_optimizer.zero_grad(set_to_none=True)
            
            with context:
                out_dict = model(**model_args)

                # The second view goes through the sample branch only; running the
                # semantic-proxy branch again would update the accumulator twice for the
                # same batch.
                if opt.loss == 'proxy_guided_simdino_peft':
                    if 'image_2' not in input_dict:
                        raise RuntimeError(
                            'the training batch has no image_2; check the '
                            'multi-view augmentation setting.')
                    view2_out = model(
                        x=input_dict['image_2'].to(
                            opt.device, non_blocking=True),
                        labels=None,
                        anchor=None,
                        warmup=warmup_active,
                        compute_proxies=False)
                    loss_args['batch_view2'] = view2_out['embeds']
                else:
                    loss_args['batch_view2'] = None
                
                loss_args['batch'] = out_dict['embeds']
                loss_args['labels'] = class_labels
                loss_args['proxies'] = out_dict['proxies']
                loss_args['nir_proxies'] = out_dict.get('nir_proxies')
                loss_args['rc_features'] = out_dict.get('rc_features')
                loss_args['backbone_features'] = out_dict.get(
                    'backbone_features')
                loss_args['reference_backbone_features'] = out_dict.get(
                    'reference_backbone_features')
                loss_args['head_relation_embeddings'] = out_dict.get(
                    'head_relation_embeddings')
                loss_args['reference_head_embeddings'] = out_dict.get(
                    'reference_head_embeddings')
                if opt.loss in (
                        'prrd_proxy_peft', 'relation_kd_proxy_peft',
                        'relation_frob_budget_proxy_peft',
                        'head_relation_budget_proxy_peft'):
                    loss_args['reference_warmup_active'] = warmup_active
                
                loss = criterion(**loss_args)


            if not torch.isfinite(loss).all():
                raise FloatingPointError(
                    'Non-finite training loss at epoch {}, batch {} '
                    '(amp_dtype={}).'.format(
                        epoch + 1, i, opt.amp_dtype))
            epoch_loss_sum += float(loss.detach().float().item())
            epoch_loss_count += 1

            if opt.loss_nir_gradient_diagnostics and i == 0:
                log_nir_gradient_diagnostics(model, criterion, epoch)
            
            refresh_progress = (
                i == 0 or (i + 1) % opt.progress_update_interval == 0 or
                i + 1 == len(data_iterator))
            if refresh_progress:
                if (hasattr(criterion, 'last_mcr2_score')
                        and criterion.last_mcr2_score is not None):
                    rc_start = rc_end = 0.0
                    if criterion.last_rc_scores is not None:
                        rc_start = criterion.last_rc_scores[0].item()
                        rc_end = criterion.last_rc_scores[-1].item()
                    data_iterator.set_postfix_str(
                        ('Loss: {:.4f} PA: {:.3f} RqN: {:.3f} '
                         'MCR2N: {:.3f} RC: {:.4f} dRC: {:+.3f} '
                         'Ramp: {:.2f}').format(
                            loss.item(), criterion.last_proxy_loss.item(),
                            criterion.last_proxy_rate_score.item(),
                            criterion.last_mcr2_score.item(),
                            criterion.last_rc_loss.item(),
                            rc_end - rc_start,
                            criterion.last_aux_ramp), refresh=False)
                elif (hasattr(criterion, 'last_gradient_scale')
                        and criterion.last_reference_geometry is not None):
                    data_iterator.set_postfix_str(
                        ('Loss: {:.4f} PA: {:.3f} HRel: {:.4f} '
                         'Dist: {:.3f} Scale: {:.3f} gRatio: {:.3f} '
                         'Ramp: {:.2f}').format(
                            loss.item(), criterion.last_proxy_loss.item(),
                            criterion.last_reference_geometry.item(),
                            criterion.last_prrd_score.item(),
                            criterion.last_gradient_scale,
                            criterion.last_output_gradient_ratio,
                            criterion.last_aux_ramp), refresh=False)
                elif (hasattr(criterion, 'last_reference_geometry')
                        and criterion.last_reference_geometry is not None):
                    data_iterator.set_postfix_str(
                        ('Loss: {:.4f} PA: {:.3f} Ref: {:.4f} '
                         'PRRD: {:.3f} FrobTan: {:.3f} RelMSE: {:.3f} '
                         'Ramp: {:.2f}').format(
                            loss.item(), criterion.last_proxy_loss.item(),
                            criterion.last_reference_geometry.item(),
                            criterion.last_prrd_score.item(),
                            criterion.last_tangent_normalized_distortion.item(),
                            criterion.last_relative_relation_mse.item(),
                            criterion.last_aux_ramp), refresh=False)
                elif (hasattr(criterion, 'last_view_loss')
                        and hasattr(criterion, 'last_proxy_rate')
                        and criterion.last_view_loss is not None):
                    data_iterator.set_postfix_str(
                        ('Loss: {:.4f} PA: {:.3f} View: {:.3f} Tan: {:.3f} '
                         'Rz: {:.3f} Rq: {:.3f} Ramp: {:.2f}').format(
                            loss.item(), criterion.last_proxy_loss.item(),
                            criterion.last_view_loss.item(),
                            criterion.last_tangent_loss.item(),
                            criterion.last_expansion.item(),
                            criterion.last_proxy_rate.item(),
                            criterion.last_aux_ramp), refresh=False)
                elif (hasattr(criterion, 'last_proxy_repulsion_loss')
                        and criterion.last_proxy_repulsion_loss is not None):
                    data_iterator.set_postfix_str(
                        ('Loss: {:.4f} PA: {:.4f} P-Repel: {:.4f} '
                         'MaxSim: {:.3f}').format(
                            loss.item(),
                            criterion.last_sample_proxy_loss.item(),
                            criterion.last_proxy_repulsion_loss.item(),
                            criterion.last_max_proxy_similarity.item()),
                        refresh=False)
                elif (hasattr(criterion, 'last_flow_loss')
                        and hasattr(criterion, 'last_feature_ramp')
                        and criterion.last_flow_loss is not None):
                    data_iterator.set_postfix_str(
                        ('Loss: {:.4f} Proxy: {:.4f} NIR: {:.3e} '
                         'wNIR: {:.3e} PosNLL: {:.3f} NegNLL: {:.3f} '
                         'Gap: {:.3f} Ramp: {:.2f}').format(
                            loss.item(), criterion.last_proxy_loss.item(),
                            criterion.last_flow_loss.item(),
                            criterion.last_nir_weight *
                            criterion.last_flow_loss.item(),
                            criterion.last_sample_nll.item(),
                            criterion.last_negative_nll.item(),
                            criterion.last_nir_gap.item(),
                            criterion.last_feature_ramp), refresh=False)
                elif (hasattr(criterion, 'last_proxy_loss')
                        and hasattr(criterion, 'last_expansion')
                        and criterion.last_proxy_loss is not None
                        and criterion.last_expansion is not None):
                    data_iterator.set_postfix_str(
                        'Loss: {:.4f} Proxy: {:.4f} TCR: {:.4f}'.format(
                            loss.item(), criterion.last_proxy_loss.item(),
                            criterion.last_expansion.item()), refresh=False)
                else:
                    data_iterator.set_postfix_str(
                        'Loss: {0:.4f}'.format(loss.item()), refresh=False)
            scaler.scale(loss).backward()
            scaler.step(model.optimizer)
            if hasattr(model, 'restore_frozen_bias_slices'):
                model.restore_frozen_bias_slices()
            if hasattr(model, 'enforce_bias_slice_mask'):
                model.enforce_bias_slice_mask()
            scaler.step(model.p_optimizer)
            if criterion_optimizer is not None:
                scaler.step(criterion_optimizer)
            scaler.update()

            if opt.prompt_memory_bank:
                model.bank_optim_states_save()
                model.bank_prompt_p_save()   
            
            if opt.time:
                end = time.time()
                total_time = total_time + (end - start)
                avg_time = total_time / (i+1)
                if i >= 100:
                    print('Average time per batch: {:.8f}s'.format(avg_time))
                    opt.time = False
        
        # update learning rate  
        if opt.scheduler != 'none':
            scheduler.step()
            p_scheduler.step()
        current_epoch = epoch + 1
        if current_epoch < opt.eval_start:
            continue
        if (current_epoch - opt.eval_start) % opt.eval_interval != 0:
            continue
        
        (numeric_metrics, heatmap_features,
         heatmap_labels) = compute_evaluation_metrics(
            include_training_metrics=not opt.no_train_metrics,
            label="Testing")
        
        if not auxiliary_plots_enabled:
            tsne_features = tsne_labels = current_proxies = None
        else:
            tsne_features, tsne_labels, current_proxies = compute_tsne_inputs()
        metric_plotter.update(
            current_epoch, numeric_metrics, heatmap_features, heatmap_labels,
            tsne_features, tsne_labels, current_proxies)
        refresh_experiment_log()

        current_r1 = max(
            (values[primary_metric_name] for values in numeric_metrics.values()
             if primary_metric_name in values),
            default=0.0)
        if current_r1 > best_checkpoint_r1:
            save_best_peft_state(current_epoch, current_r1)
            best_checkpoint_r1 = current_r1

        full_result_str = ''
        for evaltype in numeric_metrics.keys():
            full_result_str += 'Embed-Type: {}:\n'.format(evaltype)
            for i,(metricname, metricval) in enumerate(numeric_metrics[evaltype].items()):
                full_result_str += '{0}{1}: {2:4.4f}'.format('\n' if i>0 else '',metricname, metricval)
            
            # save maximum recall@1
            if (primary_metric_name in numeric_metrics[evaltype]
                    and numeric_metrics[evaltype][primary_metric_name] > max_r1):
                max_r1 = numeric_metrics[evaltype][primary_metric_name]
            numeric_metrics[evaltype]['max_r@1'] = max_r1
            
            # save maximum mAP
            if numeric_metrics[evaltype].keys().__contains__('MAP'):
                if numeric_metrics[evaltype]['MAP'] > max_map:
                    max_map = numeric_metrics[evaltype]['MAP']
                numeric_metrics[evaltype]['max_map'] = max_map
            
            full_result_str += '\n'
        
        print(full_result_str)
        epoch_runtime = time.time() - epoch_start_time
        print('\nTotal Epoch Runtime: {0:4.2f}s'.format(epoch_runtime))
        if stage_a_audit is not None:
            audit_arguments = (
                current_epoch,
                epoch_loss_sum / max(epoch_loss_count, 1),
                numeric_metrics,
                epoch_learning_rate,
                epoch_runtime)
            stage_a_audit.record_epoch(*audit_arguments)
        print('\n' + termcolor.colored('===============================================', 'red', 
                attrs=['bold']) + '\n')
        
    if stage_a_audit is not None:
        stage_a_audit.finalize()
    torch.cuda.empty_cache()
    del model
    del criterion
    if experiment_logger is not None:
        try:
            experiment_logger.mark_completed()
        except Exception as error:
            print("Warning: failed to finalize experiment log: {}".format(
                error))
    run_guard.close()
    return 0
    
    

if __name__ == '__main__':
    try:
        train()
    except KeyboardInterrupt:
        mark_active_interrupted(
            "training interrupted from the keyboard (KeyboardInterrupt).")
        raise
    except BaseException as error:
        mark_active_failed(error)
        raise
