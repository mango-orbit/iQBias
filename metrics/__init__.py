import copy
import contextlib
import sys

sys.path.insert(0, '..')

import faiss
import numpy as np
from sklearn.preprocessing import normalize
import torch
from tqdm import tqdm
import torch.nn.functional as F
from metrics import e_recall, c_recall, dists, rho_spectrum, h_recall
from metrics import nmi, f1, mAP_1000, c_mAP_1000, map_r
from utilities.metric_arrays import concatenate_feature_batches
import pdb


def select(metricname, opt=None):
    if 'e_recall' in metricname:
        k = int(metricname.split('@')[-1])
        return e_recall.Metric(k)
    elif 'h_recall' in metricname:
        k = int(metricname.split('@')[-1]); return h_recall.Metric(k, opt)
    elif 'c_recall' in metricname:
        k = int(metricname.split('@')[-1])
        return c_recall.Metric(k)
    elif metricname == 'nmi':
        return nmi.Metric()
    elif metricname == 'mAP_1000':
        return mAP_1000.Metric()
    elif metricname == 'c_mAP_1000':
        return c_mAP_1000.Metric()
    elif metricname == 'f1':
        return f1.Metric()
    elif 'dists' in metricname:
        mode = metricname.split('@')[-1]
        return dists.Metric(mode)
    elif 'rho_spectrum' in metricname:
        mode, feature_type = metricname.split('@')[1:]
        embed_dim = 128 if opt is None else opt.embed_dim
        return rho_spectrum.Metric(embed_dim,
                                   mode=int(mode),
                                   feature_type=feature_type,
                                   opt=opt)
    elif 'mAP_R' in metricname:
        return map_r.Metric()
    else:
        raise NotImplementedError(
            "Metric {} not available!".format(metricname))


class MetricComputer():
    def __init__(self, metric_names, opt):
        self.pars = opt
        self.pars.evaluate_on_gpu = True
        self.metric_names = metric_names
        self.list_of_metrics = [
            select(metricname, opt) for metricname in metric_names
        ]
        self.requires = [metric.requires for metric in self.list_of_metrics]
        self.requires = list(set([x for y in self.requires for x in y]))

    def update_test_data(self):
        pass

    def compute_standard(self, opt, model, dataloader, evaltypes, device, mode,
                         return_features=True, **kwargs):
        evaltypes = copy.deepcopy(evaltypes)
        if mode == 'Train':
            n_classes = opt.n_classes
        else:
            n_classes = opt.n_test_classes
            
        ###
        feature_colls = {key: [] for key in evaltypes}

        ###
        _ = model.eval()
        avg_features = []
        with torch.no_grad():
            target_labels = []
            final_iter = tqdm(dataloader,
                        bar_format='{desc}{percentage:2.2f}% [{n_fmt}/{total_fmt}, '
                                   '{elapsed}/{remaining}{postfix}]',
                        ncols=96, ascii=True, desc='[Compute %s]: ' % mode)
                        
            for idx, inp in enumerate(final_iter):
               
                context = contextlib.nullcontext()
                with context:
                    input_img, target = inp[1]['image'], inp[0]
                    target_labels.append(target.numpy())
                    out_dict = model(input_img.to(device, non_blocking=True))
                    raw_embeds = out_dict.get('embeds')
                    if (isinstance(raw_embeds, torch.Tensor)
                            and not torch.isfinite(raw_embeds).all()):
                        nan_count = torch.isnan(raw_embeds).sum().item()
                        inf_count = torch.isinf(raw_embeds).sum().item()
                        raise FloatingPointError(
                            ('Non-finite embeddings in {} batch {}: '
                             'NaN={}, Inf={}, mix_precision={}, '
                             'amp_dtype={}.').format(
                                 mode, idx, nan_count, inf_count,
                                 opt.mix_precision, opt.amp_dtype))
                    if 'multifeature' in model.name:
                        out = out_dict['embeds']
                        for evaltype in evaltypes:
                            weights = [
                                float(x) for x in evaltype.split('-')[1:]
                            ]
                            subevaltypes = evaltype.split(
                                'Combined_')[-1].split('-')[0].split('_')
                            weighted_subfeatures = [
                                weights[i] * out[subevaltype]
                                for i, subevaltype in enumerate(subevaltypes)
                            ]
                            if 'normalize' in model.name:
                                feature_colls[evaltype].append(
                                    F.normalize(
                                        torch.cat(weighted_subfeatures,
                                                  dim=-1),
                                        dim=-1).cpu().detach().numpy())
                            else:
                                feature_colls[evaltype].append(
                                    torch.cat(weighted_subfeatures, dim=-1).
                                    cpu().detach().numpy())
                    elif opt.normal:
                        # add normalize, e_recall will be same to c_recall
                        out = F.normalize(out_dict['embeds'], dim=-1)
                        feature_colls['embeds'].append(
                            out.cpu().detach().numpy())
                    else:
                        out = out_dict['embeds']
                        feature_colls['embeds'].append(
                            out.cpu().detach().numpy())
                            
                            
                    if 'avg_features' in self.requires:
                        avg_features.append(out_dict['avg_features'].cpu().
                                            detach().numpy())

            target_labels = np.concatenate(target_labels).reshape(-1, 1)

        computed_metrics = {evaltype: {} for evaltype in evaltypes}
        extra_infos = {evaltype: {} for evaltype in evaltypes}

        res = None
        faiss.omp_set_num_threads(self.pars.kernels)
        torch.cuda.empty_cache()
        if self.pars.evaluate_on_gpu:
            res = faiss.StandardGpuResources()

        needs_cosine_features = any(
            'cos' in requirement for requirement in self.requires)
        if 'avg_features' in self.requires:
            avg_features = concatenate_feature_batches(avg_features)
        already_on_gpu = False
        for evaltype in evaltypes:
            features = concatenate_feature_batches(feature_colls[evaltype])
            hyp_features = features
            features_cos = (normalize(features, axis=1)
                            if needs_cosine_features else None)

            k_closest_points = None
            k_closest_classes = None

            ### Compute k-Means.
            if 'kmeans' in self.requires:
                # Set CPU Cluster index.
                cluster_idx = faiss.IndexFlatL2(features.shape[-1])
                if res is not None:
                    cluster_idx = faiss.index_cpu_to_gpu(res, 0, cluster_idx)
                kmeans = faiss.Clustering(features.shape[-1], n_classes)
                kmeans.niter = 20
                kmeans.min_points_per_centroid = 1
                kmeans.max_points_per_centroid = 1000000000
                # Train Kmeans.
                kmeans.train(features, cluster_idx)
                centroids = faiss.vector_float_to_array(
                    kmeans.centroids).reshape(n_classes, features.shape[-1])

            if 'kmeans_cosine' in self.requires:
                # Set CPU Cluster index.
                cluster_idx = faiss.IndexFlatIP(features_cos.shape[-1])
                if res is not None:
                    cluster_idx = faiss.index_cpu_to_gpu(res, 0, cluster_idx)
                kmeans = faiss.Clustering(features_cos.shape[-1], n_classes)
                kmeans.niter = 20
                kmeans.min_points_per_centroid = 1
                kmeans.max_points_per_centroid = 1000000000
                # Train Kmeans.
                kmeans.train(features_cos, cluster_idx)
                centroids_cosine = faiss.vector_float_to_array(
                    kmeans.centroids).reshape(n_classes,
                                              features_cos.shape[-1])
                centroids_cosine = normalize(centroids, axis=1)

            ### Compute Cluster Labels.
            if 'kmeans_nearest' in self.requires:
                faiss_search_index = faiss.IndexFlatL2(centroids.shape[-1])
                if res is not None:
                    faiss_search_index = faiss.index_cpu_to_gpu(
                        res, 0, faiss_search_index)
                faiss_search_index.add(centroids)
                _, computed_cluster_labels = faiss_search_index.search(
                    features, 1)
                
            if 'kmeans_nearest_cosine' in self.requires:
                faiss_search_index = faiss.IndexFlatIP(
                    centroids_cosine.shape[-1])
                if res is not None:
                    faiss_search_index = faiss.index_cpu_to_gpu(
                        res, 0, faiss_search_index)
                faiss_search_index.add(centroids_cosine)
                _, computed_cluster_labels_cosine = faiss_search_index.search(
                    features_cos, 1)

            ### Compute Nearest Neighbours.
            needs_nearest_features = any([
                'nearest_features' in x for x in self.requires
                if 'cos_' not in x
            ])
            
            if needs_nearest_features:
                faiss_search_index = faiss.IndexFlatL2(features.shape[-1])
                if res is not None:
                    faiss_search_index = faiss.index_cpu_to_gpu(
                        res, 0, faiss_search_index)
                faiss_search_index.add(features)

                max_kval = np.max([
                    int(x.split('@')[-1]) for x in self.requires
                    if 'nearest_features' in x and 'cos' not in x
                ])
                _, k_closest_points = faiss_search_index.search(
                    features, int(max_kval + 1))
                k_closest_classes = target_labels.reshape(-1)[
                    k_closest_points[:, 1:]]
                
                
            needs_cos_nearest_features = any([
                'nearest_features' in x for x in self.requires if 'cos_' in x
            ])
            if needs_cos_nearest_features:
                faiss_search_index = faiss.IndexFlatIP(features_cos.shape[-1])
                if res is not None:
                    faiss_search_index = faiss.index_cpu_to_gpu(
                        res, 0, faiss_search_index)
                faiss_search_index.add(features_cos)

                max_kval = np.max([
                    int(x.split('@')[-1]) for x in self.requires
                    if 'nearest_features' in x and 'cos' in x
                ])
                _, k_closest_points_cos = faiss_search_index.search(
                    features_cos, int(max_kval + 1))
                k_closest_classes_cos = target_labels.reshape(-1)[
                    k_closest_points_cos[:, 1:]]

            if self.pars.evaluate_on_gpu:
                features = torch.from_numpy(features).to(self.pars.device)
                if features_cos is not None:
                    features_cos = torch.from_numpy(features_cos).to(
                        self.pars.device)
                if 'avg_features' in self.requires and not already_on_gpu:
                    avg_features = torch.from_numpy(avg_features).to(
                        self.pars.device)
                    already_on_gpu = True

            for metric in self.list_of_metrics:
                input_dict = {}
                if 'features' in metric.requires: input_dict['features'] = features
                if 'embeds' in metric.requires: input_dict['embeds'] = features
                if 'avg_features' in metric.requires:
                    input_dict['avg_features'] = avg_features
                if 'target_labels' in metric.requires:
                    input_dict['target_labels'] = target_labels
                if 'hyp_features' in metric.requires:
                    input_dict['hyp_features'] = hyp_features
                if 'kmeans' in metric.requires:
                    input_dict['centroids'] = centroids
                if 'kmeans_nearest' in metric.requires:
                    input_dict['computed_cluster_labels'] = computed_cluster_labels

                needs_nearest_features = any([
                    'nearest_features' in x and 'cos' not in x
                    for x in metric.requires
                ])
                if needs_nearest_features:
                    metric_max_k = np.max([
                        int(x.split('@')[-1]) for x in metric.requires
                        if 'nearest_features' in x and 'cos' not in x
                    ])
                    input_dict['k_closest_classes'] = k_closest_classes[:, :metric_max_k]
                
                needs_cos_nearest_features = any([
                    'nearest_features' in x and 'cos' in x
                    for x in metric.requires
                ])
                if needs_cos_nearest_features:
                    metric_max_k = np.max([
                        int(x.split('@')[-1]) for x in metric.requires
                        if 'nearest_features' in x and 'cos' in x
                    ])
                    input_dict[
                        'k_closest_classes_cos'] = k_closest_classes_cos[:, :metric_max_k]

                computed_metrics[evaltype][metric.name] = metric(**input_dict)
            
            
            if return_features:
                info = {
                    'features': features,
                    'target_labels': target_labels,
                }
                if k_closest_classes is not None:
                    info['k_closest_classes'] = k_closest_classes
                if k_closest_points is not None:
                    image_path = np.array(dataloader.dataset.image_paths)
                    path, _ = np.split(image_path, 2, axis=1)
                    info['image_paths'] = path
                    info['retrieved'] = path[k_closest_points[:, 1:10]]
                extra_infos[evaltype] = info

        torch.cuda.empty_cache()

        return computed_metrics, extra_infos
