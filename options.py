"""Command-line options for the query/value refinement recipe.

Only the options that the refinement recipe and an end-to-end deep metric
learning run actually need are exposed.  Every switch belonging to a retired
experimental family is frozen in ``RETIRED_DEFAULTS``:

    NIR flow            SimDINO / PGDR-SimPEFT      RateFit / RC-BitFit
    PRRD / relation KD  GainNorm / Head-aware       SCR-BitFit
    Rate-Projection     adapters                    quick-experiment subsets
    t-SNE plots         legacy LR schedules         NIR CNN backbones

They are not gone from the option namespace.  The architecture and the audit
helpers still read several of them, and a YAML configuration can still override
any of them, because configuration values are written onto the namespace after
parsing.  What is gone is the command-line surface: the flags no longer exist,
so the experiments they enabled can no longer be started by accident.

The nine builder functions below keep their names and their ``return parser``
contract, because every external audit script composes the parser by calling
them by name.
"""

RETIRED_DEFAULTS = {
    'evaltypes': ['embeds'],
    'store_improvements': False,
    'save_step': 0,
    'save_results': False,
    'savename': 'group_plus_seed',
    'tsne_num_classes': 6,
    'tsne_samples_per_class': 10,
    'lr_reduce_rate': 0.5,
    'lr_reduce_step': 5,
    'lr_reduce_multi_steps': [20, 45, 70],
    'lr_linear_start': 1.0,
    'lr_linear_end': 0.01,
    'lr_linear_length': 20,
    'hyp_metric': False,
    'hyp_c': 0.1,
    'hyp_tau': 0.2,
    'hyp_clip_r': 2.3,
    'not_pretrained': False,
    'cnn_pretrained_checkpoint': '',
    'cnn_freeze_bn': False,
    'cnn_normalize_embeddings': False,
    'cnn_double_pool': False,
    'cnn_lp_pool': 0,
    'bias_layers': 'all',
    'fewshot_manifest': '',
    'beft_stage_b_output': '',
    'beft_stage_c_output': '',
    'vit_gainnorm_bitfit': False,
    'bitfit_gain_manifest': '',
    'vit_headaware_gainnorm_bitfit': False,
    'headaware_gainnorm_output': '',
    'headaware_calibration_batches': 32,
    'headaware_hutchinson_probes': 2,
    'headaware_min_multiplier': 0.5,
    'headaware_max_multiplier': 2.0,
    'rate_projection_mode': 'none',
    'rate_projection_floor': 0.95,
    'rate_projection_ema_beta': 0.99,
    'rate_projection_sketch_dim': 128,
    'rate_projection_eps_sq': 0.5,
    'rate_projection_jitter': 1e-06,
    'bitfit_scope': 'all',
    'vit_selective_bitfit': False,
    'bitfit_selection_manifest': '',
    'bitfit_diagnostics': 0,
    'bitfit_save_peft_checkpoint': 0,
    'vit_rc_bitfit': False,
    'rc_bitfit_start_layer': 9,
    'adapter': False,
    'adapter_option': 'parallel',
    'adapter_num_blocks': 12,
    'adapter_bottleneck_dim': 64,
    'adapter_scalar': 1.0,
    'adapter_ln_position': 'post',
    'adapter_init_option': 'lora',
    'simple_proxies': False,
    'simple_proxies_vpt': False,
    'subset_experiment_enabled': 0,
    'quick_train_class_count': 50,
    'quick_test_class_count': 25,
    'subset_experiment_protocol': '',
    'time': False,
    'loss_nir_w_align': 0.0075,
    'loss_nir_pa_weight': None,
    'loss_nir_proxy_lrmulti': 4000.0,
    'loss_nir_lrmulti': 50.0,
    'loss_nir_logmatch': 1.0,
    'loss_nir_noise': 0.0,
    'loss_nir_dequantization_noise': 0.0,
    'loss_nir_nf_cond_mode': 'dense',
    'loss_nir_nf_fc_depth': 8,
    'loss_nir_nf_fc_width': 128,
    'loss_nir_nf_cblocks': 5,
    'loss_nir_nf_fc_dropout': 0.0,
    'loss_nir_nf_clamp_alpha': 2.0,
    'loss_nir_nf_fc_init': 'none',
    'loss_nir_aux_weight': 0.1,
    'loss_nir_feature_ramp_epochs': 5,
    'loss_nir_transform': 'softplus',
    'loss_nir_relative_negatives': 1,
    'loss_nir_relative_margin': 0.2,
    'loss_nir_relative_temperature': 1.0,
    'loss_nir_condition_source': 'base',
    'loss_nir_condition_gradient': 0,
    'loss_nir_separate_optim': 'adam',
    'loss_nir_gradient_diagnostics': 0,
    'loss_simdino_eps': 0.05,
    'loss_simdino_compression_weight': 1.0,
    'loss_simdino_expansion_weight': 1.0,
    'loss_simdino_jitter': 1e-06,
    'simdino_peft_multiview': 0,
    'loss_simpeft_pa_weight': 1.0,
    'loss_simpeft_view_weight': 0.5,
    'loss_simpeft_tangent_weight': 0.25,
    'loss_simpeft_sample_rate_weight': 0.1,
    'loss_simpeft_proxy_rate_weight': 0.05,
    'loss_simpeft_proxy_rate_scope': 'all',
    'loss_simpeft_eps': 0.05,
    'loss_simpeft_jitter': 1e-06,
    'loss_simpeft_aux_warmup_epochs': 5,
    'loss_simpeft_tangent_min_norm': 0.0001,
    'loss_ratefit_pa_weight': 1.0,
    'loss_ratefit_proxy_rate_weight': 0.0,
    'loss_ratefit_mcr2_weight': 0.0,
    'loss_ratefit_rc_weight': 0.0,
    'loss_ratefit_eps_sq': 0.5,
    'loss_ratefit_jitter': 1e-06,
    'loss_ratefit_aux_warmup_epochs': 5,
    'loss_ratefit_rc_margin': 0.002,
    'loss_ratefit_rc_temperature': 0.01,
    'loss_prrd_weight': 1.0,
    'loss_prrd_budget': 0.05,
    'loss_prrd_eps_sq': 0.5,
    'loss_prrd_jitter': 1e-06,
    'loss_prrd_ramp_epochs': 5,
    'loss_relation_kd_weight': 1.0,
    'loss_head_relation_budget': 0.05,
    'loss_head_relation_topk': 8,
    'loss_head_relation_gradient_ratio': 0.1,
    'loss_head_relation_ramp_epochs': 5,
    'loss_proxy_repulsion_scope': 'all',
    'loss_proxy_repulsion_weight': 0.1,
    'loss_proxy_repulsion_alpha': 32.0,
    'loss_proxy_repulsion_delta': 0.0,
    'loss_proxy_repulsion_chunk_size': 1024,
    'data_ssl_set': False,
}


def basic_training_parameters(parser):
    """Data, optimization, backbone and refinement options."""

    parser.add_argument('--name', default='default', type=str,
                        help='Name of the experiment.')
    parser.add_argument('--dataset', default='cub200', type=str,
                        help='Dataset to use: cub200, cars196, sop')

    parser.add_argument('--no_train_metrics', action='store_true',
                        help='Flag. If set, no training metrics are computed and logged.')
    parser.add_argument('--no_val_metrics', action='store_true',
                        help='Flag. If set, no val metrics are computed and logged.')
    parser.add_argument('--no_test_metrics', action='store_true',
                        help='Flag. If set, no test metrics are computed and logged.')

    parser.add_argument('--evaluation_metrics', nargs='+',
                        default=['e_recall@1', 'e_recall@2', 'e_recall@4',
                                 'e_recall@8', 'nmi', 'f1', 'mAP_R'],
                        type=str,
                        help='Metrics to evaluate performance by.')
    parser.add_argument('--storage_metrics', nargs='+', default=['e_recall@1'],
                        type=str,
                        help='Improvement in these metrics will trigger checkpointing.')

    parser.add_argument('--gpu', default=[0], nargs='+', type=int,
                        help='gpus')
    parser.add_argument('--source_path', default='../dml_data', type=str,
                        help='Path to training data.')
    parser.add_argument('--save_path', default='../Results', type=str,
                        help='Where to save everything.')
    parser.add_argument('--config', default='', type=str)

    parser.add_argument('--n_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--kernels', default=14, type=int,
                        help='Number of workers for pytorch dataloader.')
    parser.add_argument('--progress_update_interval', default=10, type=int,
                        help='Refresh batch loss diagnostics every N batches.')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Mini-Batchsize to use.')
    parser.add_argument('--seed', default=0, type=int,
                        help='Random seed for reproducibility.')
    parser.add_argument('--sdg_momentum', default=0.9, type=float)

    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning Rate for network parameters.')
    parser.add_argument('--lr_proxy', default=0.5, type=float,
                        help='Learning Rate for proxies.')
    parser.add_argument('--scheduler', default='cosine', type=str,
                        help='Type of learning rate scheduling: step, multi, linear or cosine.')
    parser.add_argument('--lr_cosine_length', default=50, type=int)
    parser.add_argument('--lr_cosine_min', default=1e-6, type=float)
    parser.add_argument('--decay', default=0.0, type=float,
                        help='Weight decay for optimizer.')

    parser.add_argument('--augmentation', default='base', type=str,
                        help='image augmentation: base, big, adv, v2, auto')
    parser.add_argument('--warmup', default=0, type=int,
                        help=('Detach the sample feature extractor for the first '
                              'epochs and train the metric head and the proxies first.'))

    parser.add_argument('--optim', default='sgd', type=str,
                        help='Optimizer to use. adam, adamw, sgd')
    parser.add_argument('--loss', default='proxy_anchor', type=str,
                        help='Training objective to use. See folder <criteria> for available methods.')
    parser.add_argument('--batch_mining', default='distance', type=str,
                        help='Batchmining method to use. See folder <batchminer> for available methods.')

    parser.add_argument('--embed_dim', default=384, type=int)
    parser.add_argument(
        '--arch', default='qv_vit', type=str,
        help='Architecture entry point: vit or qv_vit.')
    parser.add_argument('--pretrained_model', default='vit_small_patch16_224_in21k',
                        type=str)
    parser.add_argument(
        '--pretrained_checkpoint', default='', type=str,
        help=('Optional local timm-format backbone checkpoint. When set, '
              'skip online pretrained-weight downloading.'))
    parser.add_argument('--model_input_size', default=224, type=int,
                        help='Square image size passed to timm and dataset transforms.')
    parser.add_argument('--pooling', default='token',
                        help='avg, token or combined')

    parser.add_argument('--full_fine_tune', action='store_true')
    parser.add_argument('--train_head', action='store_true',
                        help='fine turn the head parameters')
    parser.add_argument('--all_bias_bitfit', action='store_true',
                        help='activate legacy all-bias BitFit')
    parser.add_argument(
        '--bias_mask', default='auto', type=str,
        choices=['auto', 'linear', 'bitfit', 'q', 'k',
                 'v', 'v_out'],
        help=('Backbone bias tuning mode. auto preserves legacy all_bias_bitfit '
              'behaviour; BEFT modes select a fused attention Q/K/V bias '
              'slice, with v_out also training attn.proj.bias. The '
              'query/value slice is selected with the "qv" mode '
              'used by a YAML configuration.'))

    parser.add_argument(
        '--context_rank', default=0, type=int,
        help=('Rank of the shared image-conditioned query/value bias residual. '
              'Zero disables contextual refinement.'))
    parser.add_argument(
        '--context_targets', default='qv', type=str,
        choices=['q', 'v', 'qv'],
        help='Dynamic bias targets used when context_rank is positive.')
    parser.add_argument(
        '--head_scale', action='store_true',
        help=('Learn one positive attention-logit scale per layer and head '
              'on top of static query/value bias.'))

    parser.add_argument(
        '--beft_stage_a_output', default='', type=str,
        help=('Optional JSON output path for the parameter-accounting audit of '
              'the selected bias slices.'))

    parser.add_argument('--visual_prompt', action='store_true',
                        help='activate shared visual prompts (VPT arm)')
    parser.add_argument('--semantic_proxy', action='store_true',
                        help='activate the semantic-proxy branch (VPTSP arm)')
    parser.add_argument('--num_prompt', default=10, type=int)
    parser.add_argument('--prompt_depth', default=12, type=int)
    parser.add_argument('--prompt_init_scale', default=1.0, type=float,
                        help='Scale applied after Xavier initialization of the prompts.')
    parser.add_argument('--prompt_lr_scale', default=1.0, type=float,
                        help='Learning-rate multiplier for prompt parameters.')
    parser.add_argument('--bitfit_lr_scale', default=1.0, type=float,
                        help=('Learning-rate multiplier for frozen-backbone bias '
                              'parameters. The default 1.0 preserves the original '
                              'optimizer.'))
    parser.add_argument('--num_p_prompt', default=5, type=int)
    parser.add_argument('--p_prompt_depth', default=3, type=int)
    parser.add_argument('--proxy_net_depth', default=12, type=int,
                        help='depth of the proxy net')
    parser.add_argument('--proxy_net_head', default='isolated', type=str,
                        help='shared or isolated')
    parser.add_argument('--proxy_mix_bias', action='store_true',
                        help='mix original proxies into vpts proxies')
    parser.add_argument('--proxy_mix_keep_ratio', default=0.8, type=float)
    parser.add_argument('--proxy_bias_ratio', default=0.5, type=float)
    parser.add_argument('--prompt_block_step', default=0, type=int)
    parser.add_argument('--prompt_p_block_step', default=0, type=int)
    parser.add_argument('--proxy_data_range', default='in_batch', type=str,
                        help=('in_batch: only optimize proxies inside batch; '
                              'both_all: optimize all in batch, positive and '
                              'negative in all proxies.'))
    parser.add_argument('--semantic_mix_type', default='gru_relu', type=str,
                        help='mix, gru ...')
    parser.add_argument('--rnn_dropout', default=0.0, type=float)
    parser.add_argument('--feature_dropout', default=0.0, type=float)

    parser.set_defaults(**RETIRED_DEFAULTS)
    return parser


def scale_optimizing_parameters(parser):
    """Precision, evaluation cadence and reporting options."""

    parser.add_argument('--mix_precision', action='store_true',
                        help='activate mix precision')
    parser.add_argument(
        '--amp_dtype', default='float16', type=str,
        choices=['float16', 'bfloat16'],
        help='CUDA autocast dtype.')
    parser.add_argument(
        '--cudnn_benchmark', action='store_true',
        help=('Enable cuDNN convolution autotuning. The first batch for each '
              'new input shape is slower while cuDNN selects and caches a '
              'fast algorithm; fixed-shape training can then run faster. '
              'Deterministic convolution selection remains disabled in both '
              'legacy and benchmark modes.'))
    parser.add_argument('--world_size', default=1, type=int, help='number of gpus')
    parser.add_argument('--prompt_memory_bank', action='store_true',
                        help='active prompt memory bank')
    parser.add_argument('--normal', action='store_true',
                        help='normalize in testing')
    parser.add_argument('--checkpoint', action='store_true',
                        help='active gpu checkpoint')
    parser.add_argument('--show_trainable_pars', action='store_true',
                        help='show trainable parameters')
    parser.add_argument('--eval_start', default=0, type=int,
                        help='start epoch for evaluation')
    parser.add_argument('--eval_interval', default=1, type=int,
                        help='evaluate once every N epochs after eval_start')
    parser.add_argument('--eval_before_training', default=0, type=int,
                        choices=[0, 1],
                        help='evaluate initialized model at epoch 0: 1=yes, 0=no')
    parser.add_argument('--fast_evaluation', default=0, type=int,
                        choices=[0, 1],
                        help=('1=only compute the dataset-standard Recall '
                              'cutoffs (1/10/100/1000 for SOP, 1/2/4/8 '
                              'otherwise) and keep best-metric records; skip '
                              'F1/NMI/mAP. Plot generation is controlled by '
                              '--disable_plots.'))
    parser.add_argument('--disable_plots', default=1, type=int,
                        choices=[0, 1],
                        help=('1=skip every metric curve, heatmap, histogram '
                              'and t-SNE plot while retaining metric records.'))
    parser.add_argument('--heatmap_max_images', default=500, type=int,
                        help='number of ordered test embeddings in each heatmap')

    return parser


def nir_parameters(parser):
    """Retired: the NIR conditional normalizing flow family.

    Kept as an empty builder so that external audit scripts can still compose
    the parser by name. Its switches live in ``RETIRED_DEFAULTS``.
    """
    return parser


def simdino_parameters(parser):
    """Retired: the SimDINO and PGDR-SimPEFT coding-rate families.

    Kept as an empty builder so that external audit scripts can still compose
    the parser by name. Its switches live in ``RETIRED_DEFAULTS``.
    """
    return parser


def ratefit_parameters(parser):
    """Retired: the RateFit, MCR2 and rate-constrained BitFit families.

    Kept as an empty builder so that external audit scripts can still compose
    the parser by name. Its switches live in ``RETIRED_DEFAULTS``.
    """
    return parser


def prrd_parameters(parser):
    """Retired: the PRRD and relation-distillation families.

    Kept as an empty builder so that external audit scripts can still compose
    the parser by name. Its switches live in ``RETIRED_DEFAULTS``.
    """
    return parser


def loss_specific_parameters(parser):
    """Proxy-Anchor knob set."""

    parser.add_argument('--loss_oproxy_mode', default='anchor', type=str,
                        help='Proxy-method: anchor = ProxyAnchor, nca = ProxyNCA.')
    parser.add_argument('--loss_oproxy_lrmulti', default=200, type=float,
                        help='Learning rate multiplier for proxies.')
    parser.add_argument('--loss_oproxy_pos_alpha', default=32, type=float,
                        help='Inverted temperature/scaling for positive sample-proxy similarities.')
    parser.add_argument('--loss_oproxy_neg_alpha', default=32, type=float,
                        help='Inverted temperature/scaling for negative sample-proxy similarities.')
    parser.add_argument('--loss_oproxy_pos_delta', default=0.1, type=float,
                        help='Threshold for positive sample-proxy similarities')
    parser.add_argument('--loss_oproxy_neg_delta', default=-0.1, type=float,
                        help='Threshold for negative sample-proxy similarities')

    return parser


def batchmining_specific_parameters(parser):
    """Hyperparameters for the available batchmining methods."""

    parser.add_argument('--miner_distance_lower_cutoff', default=0.5, type=float,
                        help='Cutoff distance value below which pairs are ignored.')
    parser.add_argument('--miner_distance_upper_cutoff', default=1.4, type=float,
                        help='Cutoff distance value above which pairs are ignored.')
    parser.add_argument('--miner_rho_distance_lower_cutoff', default=0.5, type=float,
                        help='Same behaviour as with standard distance-based mining.')
    parser.add_argument('--miner_rho_distance_upper_cutoff', default=1.4, type=float,
                        help='Same behaviour as with standard distance-based mining.')
    parser.add_argument('--miner_rho_distance_cp', default=0.2, type=float,
                        help='Probability with which label assignments are flipped.')
    parser.add_argument('--miner_semihard_margin', default=0.2, type=float,
                        help='Margin value for semihard mining.')
    parser.add_argument('--internal_split', action='store_true')

    return parser


def batch_creation_parameters(parser):
    """Parameters for batch sampling methods."""

    parser.add_argument('--data_sampler', default='class_random', type=str,
                        help='Batch-creation method. Default <class_random> ensures that for each class, '
                             'at least --samples_per_class samples per class are available in each minibatch.')
    parser.add_argument('--samples_per_class', default=2, type=int,
                        help='Number of samples in one class drawn before choosing the next class. '
                             'Set to >1 for losses other than ProxyNCA.')

    return parser
