"""Update training metric curves after every evaluation epoch."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import os
import re
import shutil

import numpy as np

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill


_PYPLOT = None


def _get_pyplot():
    global _PYPLOT
    if _PYPLOT is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
        _PYPLOT = pyplot
    return _PYPLOT

class TrainingMetricPlotter:
    """Collect metrics and overwrite PDF curves and an Excel summary per update."""

    # Timestamps use a fixed UTC+8 offset rather than the host time zone, so a
    # server set to UTC still reports wall-clock time in the run directories.
    BEIJING_TIMEZONE = timezone(timedelta(hours=8), name="Asia/Shanghai")

    RECALL_METRICS = (
        ("e_recall@1", "Recall@1"),
        ("e_recall@2", "Recall@2"),
        ("e_recall@4", "Recall@4"),
        ("e_recall@8", "Recall@8"),
    )

    EXCEL_METRICS = (
        (("e_recall@1",), "Recall@1"),
        (("e_recall@2",), "Recall@2"),
        (("e_recall@4",), "Recall@4"),
        (("e_recall@8",), "Recall@8"),
        (("f1",), "F1"),
        (("nmi",), "NMI"),
        (("mAP_R", "MAP"), "mAP"),
    )

    def __init__(self, results_root, dataset_name, backbone_name,
                 heatmap_max_images=500, tsne_num_classes=6,
                 tsne_samples_per_class=10, auxiliary_plots=True,
                 plots_enabled=True, evaluation_metrics=None):
        timestamp = datetime.now(self.BEIJING_TIMEZONE).strftime(
            "%Y_%m_%d_%H_%M_%S")
        dataset_dir = self._safe_name(dataset_name)
        backbone_dir = self._safe_name(backbone_name)

        # Several GPUs may start the same dataset/backbone in the same second. An
        # atomic mkdir allocates the directory, so two processes never share a run
        # directory; collisions append _1, _2 in order.
        run_parent = Path(results_root) / dataset_dir / backbone_dir
        run_parent.mkdir(parents=True, exist_ok=True)
        suffix = 0
        while True:
            directory_name = timestamp if suffix == 0 else "{}_{}".format(
                timestamp, suffix)
            candidate = run_parent / directory_name
            try:
                candidate.mkdir(exist_ok=False)
                self.run_dir = candidate
                break
            except FileExistsError:
                suffix += 1

        self.image_dir = self.run_dir / "image"
        self.heatmap_dir = self.image_dir / "Heat_map"
        self.histogram_dir = self.image_dir / "Histogram"
        self.tsne_dir = self.image_dir / "t-SNE"
        self.proxy_heatmap_dir = self.image_dir / "Proxy_Similarity"
        self.records_dir = self.run_dir / "records"
        self.auxiliary_plots = bool(auxiliary_plots)
        self.plots_enabled = bool(plots_enabled)
        requested_recalls = []
        if evaluation_metrics is not None:
            for metric_name in evaluation_metrics:
                # Euclidean and Hyperbolic Recall share the same result-table
                # columns, while their raw metric keys remain distinct.
                match = re.fullmatch(r"(?:e|h)_recall@(\d+)", str(metric_name))
                if match:
                    requested_recalls.append(
                        (str(metric_name), "Recall@{}".format(match.group(1))))
        self.recall_metrics = tuple(requested_recalls) or self.RECALL_METRICS
        recall_excel_metrics = tuple(
            ((metric_name,), display_name)
            for metric_name, display_name in self.recall_metrics)
        auxiliary_excel_metrics = self.EXCEL_METRICS[4:]
        self.excel_metrics = recall_excel_metrics + (
            auxiliary_excel_metrics if self.auxiliary_plots else ())

        if self.plots_enabled:
            self.image_dir.mkdir(parents=True, exist_ok=True)
        if self.plots_enabled and self.auxiliary_plots:
            self.heatmap_dir.mkdir(parents=True, exist_ok=True)
            self.histogram_dir.mkdir(parents=True, exist_ok=True)
            self.tsne_dir.mkdir(parents=True, exist_ok=True)
            self.proxy_heatmap_dir.mkdir(parents=True, exist_ok=True)
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.excel_path = self.records_dir / "best_metrics.xlsx"
        self.text_path = self.records_dir / "best_metrics.txt"
        self.json_path = self.records_dir / "best_metrics.json"
        self.heatmap_max_images = int(heatmap_max_images)
        if self.heatmap_max_images <= 0:
            raise ValueError("heatmap_max_images must be greater than 0")
        self.tsne_num_classes = int(tsne_num_classes)
        self.tsne_samples_per_class = int(tsne_samples_per_class)
        if self.tsne_num_classes <= 0:
            raise ValueError("tsne_num_classes must be greater than 0")
        if self.tsne_samples_per_class <= 0:
            raise ValueError("tsne_samples_per_class must be greater than 0")
        self.epochs = []
        self.history = {}

    @staticmethod
    def _safe_name(value):
        value = str(value).strip()
        return re.sub(r"[^A-Za-z0-9._-]+", "_", value) or "unknown"

    def archive_run_config(self, config_path):
        """Store the effective YAML beside the outputs of this exact run.

        Shell launchers create effective configs in a hidden pending directory.
        Those files are moved atomically. Hand-written configs elsewhere are
        copied so source files under ``config/`` are never modified.
        """
        source_path = Path(config_path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(
                "Run configuration does not exist: {}".format(source_path))

        archived_path = self.run_dir / "run_config.yaml"
        if source_path == archived_path.resolve():
            return archived_path

        generated_directories = {".pending_run_configs", "run_configs"}
        if source_path.parent.name in generated_directories:
            os.replace(str(source_path), str(archived_path))
            try:
                source_path.parent.rmdir()
            except OSError:
                # Other pending/legacy configs may still be present.
                pass
        else:
            temporary_path = archived_path.with_name(".run_config.yaml.tmp")
            shutil.copy2(str(source_path), str(temporary_path))
            os.replace(str(temporary_path), str(archived_path))
        return archived_path

    @staticmethod
    def _as_float(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "item"):
            value = value.item()
        return float(value)

    @staticmethod
    def _flatten_metrics(numeric_metrics):
        """Merge eval types, preferring the first value for duplicate names."""
        flattened = {}
        for metrics_for_type in numeric_metrics.values():
            for name, value in metrics_for_type.items():
                flattened.setdefault(name, value)
        return flattened

    def update(self, epoch, numeric_metrics, test_features=None, test_labels=None,
               tsne_features=None, tsne_labels=None, proxy_features=None):
        """Refresh all metric records and evaluation visualizations."""
        flattened = self._flatten_metrics(numeric_metrics)

        epoch = int(epoch)
        if epoch not in self.epochs:
            self.epochs.append(epoch)

        tracked_names = [name for name, _ in self.recall_metrics]
        tracked_names.extend(["f1", "nmi", "mAP_R", "MAP"])
        for name in tracked_names:
            if name in flattened:
                self.history.setdefault(name, {})[epoch] = self._as_float(
                    flattened[name]
                )

        self._write_best_metrics()
        self._write_best_json()
        if not self.auxiliary_plots:
            self._write_best_text()
        if self.plots_enabled:
            self._plot_recall()
        if self.plots_enabled and self.auxiliary_plots:
            self._plot_single(("f1",), "F1", "f1.pdf")
            self._plot_single(("nmi",), "NMI", "nmi.pdf")
            self._plot_single(("mAP_R", "MAP"), "mAP", "mAP.pdf")
            if test_features is not None:
                similarities = self._compute_similarity_matrix(test_features)
                self._save_similarity_heatmap(epoch, similarities)
                if test_labels is not None:
                    self._save_pair_similarity_histogram(
                        epoch, similarities, test_labels)
            if proxy_features is not None:
                self._save_proxy_similarity_heatmap(epoch, proxy_features)
            if (tsne_features is not None and tsne_labels is not None and
                    proxy_features is not None):
                self._save_tsne_comparison(
                    epoch, tsne_features, tsne_labels, proxy_features)

    @staticmethod
    def _as_numpy(values, dtype=None):
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        return np.asarray(values, dtype=dtype)

    def _save_tsne_comparison(self, epoch, features, labels, proxies):
        """Plot joint 2D and spherical 3D t-SNE views in one figure."""
        plt = _get_pyplot()
        from matplotlib.lines import Line2D
        from sklearn.manifold import TSNE

        features = self._as_numpy(features, dtype=np.float32)
        labels = self._as_numpy(labels).reshape(-1)
        proxies = self._as_numpy(proxies, dtype=np.float32)
        if features.ndim != 2 or proxies.ndim != 2:
            raise ValueError("t-SNE features and proxies must be 2D arrays")
        if features.shape[0] != labels.shape[0]:
            raise ValueError("t-SNE labels must match the sample features")
        if features.shape[1] != proxies.shape[1]:
            raise ValueError("t-SNE samples and proxies must share feature dimension")

        available_classes = sorted(int(value) for value in np.unique(labels))
        selected_classes = available_classes[:self.tsne_num_classes]
        sample_parts = []
        label_parts = []
        valid_classes = []
        selected_proxies = []
        for class_id in selected_classes:
            if class_id < 0 or class_id >= proxies.shape[0]:
                continue
            class_indices = np.flatnonzero(labels == class_id)[
                :self.tsne_samples_per_class]
            if class_indices.size == 0:
                continue
            sample_parts.append(features[class_indices])
            label_parts.append(np.full(class_indices.size, class_id, dtype=np.int64))
            selected_proxies.append(proxies[class_id])
            valid_classes.append(class_id)

        if not sample_parts:
            raise ValueError("No matching class samples and proxies for t-SNE")
        sample_features = np.concatenate(sample_parts, axis=0)
        sample_labels = np.concatenate(label_parts, axis=0)
        selected_proxies = np.stack(selected_proxies, axis=0)

        combined = np.concatenate((sample_features, selected_proxies), axis=0)
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        combined = combined / np.clip(norms, 1e-12, None)
        perplexity = min(30.0, max(1.0, (combined.shape[0] - 1) / 3.0))

        def project(n_components):
            initialization = (
                "pca" if combined.shape[0] > n_components else "random")
            coordinates = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                init=initialization,
                learning_rate="auto",
                n_iter=1000,
                random_state=0,
                method="barnes_hut",
            ).fit_transform(combined)
            coordinates -= coordinates.mean(axis=0, keepdims=True)
            max_radius = np.linalg.norm(coordinates, axis=1).max()
            return coordinates / max(max_radius, 1e-12) * 0.92

        coordinates_2d = project(2)
        coordinates_3d = project(3)
        sample_count = sample_features.shape[0]
        samples_2d = coordinates_2d[:sample_count]
        proxies_2d = coordinates_2d[sample_count:]
        samples_3d = coordinates_3d[:sample_count]
        proxies_3d = coordinates_3d[sample_count:]

        fig = plt.figure(figsize=(15.2, 7.2))
        axis_2d = fig.add_subplot(1, 2, 1)
        axis_3d = fig.add_subplot(1, 2, 2, projection="3d")
        boundary = plt.Circle(
            (0.0, 0.0), 1.0, facecolor="#F5F7FA",
            edgecolor="#9AA0A6", linewidth=0.8, alpha=0.65, zorder=0)
        axis_2d.add_patch(boundary)

        azimuth = np.linspace(0.0, 2.0 * np.pi, 37)
        elevation = np.linspace(0.0, np.pi, 19)
        sphere_x = np.outer(np.cos(azimuth), np.sin(elevation))
        sphere_y = np.outer(np.sin(azimuth), np.sin(elevation))
        sphere_z = np.outer(np.ones_like(azimuth), np.cos(elevation))
        axis_3d.plot_wireframe(
            sphere_x, sphere_y, sphere_z, color="#9AA0A6",
            linewidth=0.35, alpha=0.16, rstride=2, cstride=2)

        color_map = plt.get_cmap("tab10", max(len(valid_classes), 1))
        for color_index, class_id in enumerate(valid_classes):
            class_mask = sample_labels == class_id
            color = color_map(color_index)
            axis_2d.scatter(
                samples_2d[class_mask, 0],
                samples_2d[class_mask, 1],
                marker="o", s=38, color=color, alpha=0.76,
                edgecolors="white", linewidths=0.4, zorder=2,
                label="Class {}".format(class_id))
            axis_2d.scatter(
                proxies_2d[color_index, 0],
                proxies_2d[color_index, 1],
                marker="*", s=250, color=color, alpha=1.0,
                edgecolors="black", linewidths=0.9, zorder=4)
            axis_3d.scatter(
                samples_3d[class_mask, 0],
                samples_3d[class_mask, 1],
                samples_3d[class_mask, 2],
                marker="o", s=34, color=color, alpha=0.72,
                edgecolors="white", linewidths=0.35,
                depthshade=False)
            axis_3d.scatter(
                proxies_3d[color_index, 0],
                proxies_3d[color_index, 1],
                proxies_3d[color_index, 2],
                marker="*", s=230, color=color, alpha=1.0,
                edgecolors="black", linewidths=0.85, depthshade=False)

        axis_2d.set_title("2D t-SNE")
        axis_2d.set_xlabel("t-SNE 1")
        axis_2d.set_ylabel("t-SNE 2")
        axis_2d.set_xlim(-1.05, 1.05)
        axis_2d.set_ylim(-1.05, 1.05)
        axis_2d.set_xticks([-1.0, 0.0, 1.0])
        axis_2d.set_yticks([-1.0, 0.0, 1.0])
        axis_2d.set_aspect("equal", adjustable="box")
        axis_2d.set_axisbelow(True)
        axis_2d.grid(linestyle="--", linewidth=0.6, alpha=0.28)
        axis_2d.spines["top"].set_visible(False)
        axis_2d.spines["right"].set_visible(False)

        axis_3d.set_title("3D t-SNE")
        axis_3d.set_xlabel("t-SNE 1")
        axis_3d.set_ylabel("t-SNE 2")
        axis_3d.set_zlabel("t-SNE 3")
        axis_3d.set_xlim(-1.05, 1.05)
        axis_3d.set_ylim(-1.05, 1.05)
        axis_3d.set_zlim(-1.05, 1.05)
        axis_3d.set_xticks([-1.0, 0.0, 1.0])
        axis_3d.set_yticks([-1.0, 0.0, 1.0])
        axis_3d.set_zticks([-1.0, 0.0, 1.0])
        axis_3d.set_box_aspect((1.0, 1.0, 1.0))
        axis_3d.view_init(elev=22, azim=38)

        legend_handles, _ = axis_2d.get_legend_handles_labels()
        legend_handles.append(Line2D(
            [0], [0], marker="*", linestyle="none", markersize=12,
            markerfacecolor="#7F7F7F", markeredgecolor="black",
            label="Class Proxy"))
        fig.suptitle(
            "t-SNE of Samples and Class Proxies (Epoch {})".format(int(epoch)),
            fontsize=14)
        fig.legend(
            handles=legend_handles, loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(4, len(legend_handles)), fontsize=8)
        fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.95))

        filename = "{}_t_SNE.pdf".format(int(epoch))
        output_path = self.tsne_dir / filename
        temporary_path = self.tsne_dir / ".{}.tmp".format(filename)
        fig.savefig(temporary_path, format="pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)
        os.replace(str(temporary_path), str(output_path))


    def _save_proxy_similarity_heatmap(self, epoch, proxy_features):
        """Plot cosine similarity matrix of the first 100 class proxies."""
        plt = _get_pyplot()
        proxies = self._as_numpy(proxy_features, dtype=np.float32)
        # Take the first 100 classes (CUB200 official train split = 100 classes)
        max_proxies = min(100, proxies.shape[0])
        proxies = proxies[:max_proxies]
        # Normalize and compute cosine similarity
        norms = np.linalg.norm(proxies, axis=1, keepdims=True)
        normalized = proxies / np.clip(norms, 1e-12, None)
        similarities = np.clip(normalized @ normalized.T, -1.0, 1.0)

        fig, axis = plt.subplots(figsize=(8.5, 7.2))
        image = axis.imshow(
            similarities,
            cmap="viridis",
            vmin=-1.0,
            vmax=1.0,
            interpolation="nearest",
            aspect="equal",
            rasterized=True,
        )
        axis.set_title(
            "Class Proxy Cosine Similarity - First {} Classes (Epoch {})".format(
                max_proxies, int(epoch)))
        axis.set_xlabel("Proxy Class Index")
        axis.set_ylabel("Proxy Class Index")

        tick_step = max(1, max_proxies // 5)
        ticks = list(range(0, max_proxies, tick_step))
        if ticks and ticks[-1] != max_proxies - 1:
            ticks.append(max_proxies - 1)
        axis.set_xticks(ticks)
        axis.set_xticklabels([str(t) for t in ticks])
        axis.set_yticks(ticks)
        axis.set_yticklabels([str(t) for t in ticks])
        colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label("Cosine Similarity")
        fig.tight_layout()

        filename = "{}_proxy_similarity.pdf".format(int(epoch))
        output_path = self.proxy_heatmap_dir / filename
        temporary_path = self.proxy_heatmap_dir / ".{}.tmp".format(filename)
        fig.savefig(temporary_path, format="pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)
        os.replace(str(temporary_path), str(output_path))

    def _compute_similarity_matrix(self, features):
        """Compute cosine similarity once for all similarity visualizations."""
        if hasattr(features, "detach"):
            features = features.detach().cpu().numpy()
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("test_features must be a non-empty 2D array")

        sample_count = min(self.heatmap_max_images, features.shape[0])
        features = features[:sample_count]
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        normalized = features / np.clip(norms, 1e-12, None)
        return np.clip(normalized @ normalized.T, -1.0, 1.0)

    def _save_similarity_heatmap(self, epoch, similarities):
        """Plot the precomputed cosine similarity matrix."""
        plt = _get_pyplot()
        sample_count = similarities.shape[0]
        fig, axis = plt.subplots(figsize=(8.5, 7.2))
        image = axis.imshow(
            similarities,
            cmap="viridis",
            vmin=-1.0,
            vmax=1.0,
            interpolation="nearest",
            aspect="equal",
            rasterized=True,
        )
        axis.set_title(
            f"Test Embedding Cosine Similarity (Epoch {int(epoch)})")
        axis.set_xlabel("Test Image Index")
        axis.set_ylabel("Test Image Index")

        tick_step = max(1, sample_count // 5)
        ticks = list(range(0, sample_count, tick_step))
        if ticks[-1] != sample_count - 1:
            ticks.append(sample_count - 1)
        axis.set_xticks(ticks)
        axis.set_xticklabels([tick + 1 for tick in ticks])
        axis.set_yticks(ticks)
        axis.set_yticklabels([tick + 1 for tick in ticks])
        colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label("Cosine Similarity")
        fig.tight_layout()

        filename = f"{int(epoch)}_heatmap.pdf"
        output_path = self.heatmap_dir / filename
        temporary_path = self.heatmap_dir / f".{filename}.tmp"
        fig.savefig(temporary_path, format="pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)
        os.replace(str(temporary_path), str(output_path))

    def _save_pair_similarity_histogram(self, epoch, similarities, labels):
        """Plot normalized positive/negative pair distributions in 0.01 bins."""
        plt = _get_pyplot()
        if hasattr(labels, "detach"):
            labels = labels.detach().cpu().numpy()
        labels = np.asarray(labels).reshape(-1)
        sample_count = similarities.shape[0]
        if labels.shape[0] < sample_count:
            raise ValueError("test_labels must match the sampled test features")
        labels = labels[:sample_count]

        row_indices, column_indices = np.triu_indices(sample_count, k=1)
        pair_similarities = (
            similarities[row_indices, column_indices] + 1.0) / 2.0
        positive_mask = labels[row_indices] == labels[column_indices]
        positive_similarities = pair_similarities[positive_mask]
        negative_similarities = pair_similarities[~positive_mask]

        bin_edges = np.linspace(0.0, 1.0, 101)
        positive_counts, _ = np.histogram(positive_similarities, bins=bin_edges)
        negative_counts, _ = np.histogram(negative_similarities, bins=bin_edges)
        positive_ratios = positive_counts.astype(np.float64) / max(
            positive_similarities.size, 1)
        negative_ratios = negative_counts.astype(np.float64) / max(
            negative_similarities.size, 1)

        centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        bar_width = 0.01 * 0.92
        fig, axis = plt.subplots(figsize=(9.0, 5.8))
        axis.bar(
            centers, negative_ratios, width=bar_width,
            color="#D55E00", edgecolor="#D55E00", linewidth=0.25,
            alpha=0.50, zorder=2,
            label=f"Negative pairs (n={negative_similarities.size})")
        axis.bar(
            centers, positive_ratios, width=bar_width,
            color="#0072B2", edgecolor="#0072B2", linewidth=0.25,
            alpha=0.58, zorder=3,
            label=f"Positive pairs (n={positive_similarities.size})")

        axis.set_title(
            f"Positive/Negative Pair Similarity Distribution (Epoch {int(epoch)})")
        axis.set_xlabel("Normalized Cosine Similarity")
        axis.set_ylabel("Pair Proportion per 0.01 Bin")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xticks(np.arange(0.0, 1.01, 0.2))
        axis.set_xticks(np.arange(0.0, 1.001, 0.05), minor=True)
        axis.set_yticks(np.arange(0.0, 1.01, 0.1))
        axis.set_axisbelow(True)
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(loc="best")
        fig.tight_layout()

        filename = f"{int(epoch)}_histogram.pdf"
        output_path = self.histogram_dir / filename
        temporary_path = self.histogram_dir / f".{filename}.tmp"
        fig.savefig(temporary_path, format="pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)
        os.replace(str(temporary_path), str(output_path))

    def _write_best_metrics(self):
        """Store best-so-far values and their source epochs for every test epoch."""
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Best Metrics"

        headers = ["Test Epoch"]
        for _, display_name in self.excel_metrics:
            headers.extend([f"{display_name} Best", f"{display_name} Best Epoch"])
        sheet.append(headers)

        for test_epoch in sorted(self.epochs):
            row = [test_epoch]
            for aliases, _ in self.excel_metrics:
                values = {}
                for metric_name in aliases:
                    values.update(self.history.get(metric_name, {}))
                eligible_epochs = [epoch for epoch in values if epoch <= test_epoch]
                if eligible_epochs:
                    best_epoch = max(
                        eligible_epochs,
                        key=lambda epoch: (values[epoch], -epoch),
                    )
                    row.extend([values[best_epoch], best_epoch])
                else:
                    row.extend([None, None])
            sheet.append(row)

        header_fill = PatternFill("solid", fgColor="1F4E78")
        for cell in sheet[1]:
            cell.font = Font(color="FFFFFF", bold=True)
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center")

        for row in sheet.iter_rows(min_row=2):
            for column_index, cell in enumerate(row, start=1):
                cell.alignment = Alignment(horizontal="center")
                if column_index > 1 and column_index % 2 == 0 and cell.value is not None:
                    cell.number_format = "0.000000"

        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        sheet.column_dimensions["A"].width = 13
        for column in sheet.iter_cols(min_col=2, max_col=sheet.max_column):
            sheet.column_dimensions[column[0].column_letter].width = 22

        temporary_path = self.records_dir / ".best_metrics.tmp.xlsx"
        workbook.save(temporary_path)
        workbook.close()
        os.replace(str(temporary_path), str(self.excel_path))

    def _write_best_text(self):
        """Write a compact, human-readable best Recall summary."""
        latest_epoch = max(self.epochs)
        lines = ["Latest Test Epoch: {}".format(latest_epoch)]
        for aliases, display_name in self.excel_metrics:
            values = {}
            for metric_name in aliases:
                values.update(self.history.get(metric_name, {}))
            eligible_epochs = [
                epoch for epoch in values if epoch <= latest_epoch]
            if not eligible_epochs:
                continue
            best_epoch = max(
                eligible_epochs,
                key=lambda epoch: (values[epoch], -epoch))
            lines.append("{} Best: {:.6f} (Epoch {})".format(
                display_name, values[best_epoch], best_epoch))

        temporary_path = self.records_dir / ".best_metrics.tmp.txt"
        with open(temporary_path, "w", encoding="utf-8") as output_file:
            output_file.write("\n".join(lines) + "\n")
        os.replace(str(temporary_path), str(self.text_path))

    def _write_best_json(self):
        """Write a small exact-value record for fast cross-run indexing."""
        latest_epoch = max(self.epochs)
        payload = {
            "latest_test_epoch": int(latest_epoch),
            "metrics": {},
            # Raw values of every evaluated epoch are kept so that comparisons stay
            # exact at the same seed and epoch, instead of comparing an early candidate
            # value against a baseline best-so-far.
            "history": {},
        }
        for aliases, display_name in self.excel_metrics:
            values = {}
            for metric_name in aliases:
                values.update(self.history.get(metric_name, {}))
            eligible_epochs = [
                epoch for epoch in values if epoch <= latest_epoch]
            if not eligible_epochs:
                continue
            best_epoch = max(
                eligible_epochs,
                key=lambda epoch: (values[epoch], -epoch))
            payload["metrics"][display_name] = {
                "best": float(values[best_epoch]),
                "epoch": int(best_epoch),
            }
            payload["history"][display_name] = [
                {"epoch": int(epoch), "value": float(values[epoch])}
                for epoch in sorted(values)
            ]

        temporary_path = self.records_dir / ".best_metrics.tmp.json"
        with open(temporary_path, "w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
        os.replace(str(temporary_path), str(self.json_path))


    def _plot_recall(self):
        series = []
        for metric_name, display_name in self.recall_metrics:
            values = self.history.get(metric_name, {})
            if values:
                series.append((display_name, values))
        if series:
            self._save_plot(series, "Recall", "recall.pdf")

    def _plot_single(self, metric_names, display_name, filename):
        for metric_name in metric_names:
            values = self.history.get(metric_name, {})
            if values:
                self._save_plot([(display_name, values)], display_name, filename)
                return

    def _save_plot(self, series, title, filename):
        plt = _get_pyplot()
        fig, axis = plt.subplots(figsize=(8.5, 5.5))

        all_epochs = sorted({epoch for _, values in series for epoch in values})
        for index, (display_name, values) in enumerate(series):
            epochs = sorted(values)
            scores = [values[epoch] for epoch in epochs]
            axis.plot(epochs, scores, marker="o", linewidth=1.8, label=display_name)

            best_epoch = max(epochs, key=lambda item: values[item])
            best_value = values[best_epoch]
            axis.scatter(
                [best_epoch], [best_value], color="red", s=42, zorder=5
            )
            axis.annotate(
                f"best={best_value:.4f}\nepoch={best_epoch}",
                (best_epoch, best_value),
                xytext=(5, 8 + index * 7),
                textcoords="offset points",
                color="red",
                fontsize=8,
            )

        axis.set_title(f"{title} vs. Epoch")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Score")
        axis.set_xticks(all_epochs)
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.legend(loc="best")
        fig.tight_layout()

        output_path = self.image_dir / filename
        temporary_path = self.image_dir / f".{filename}.tmp"
        fig.savefig(temporary_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        os.replace(str(temporary_path), str(output_path))
