"""Deterministic class slicing for the lightweight experiment protocol.

This module deliberately does not change the official train/test boundary.
It only keeps a leading prefix *inside* each split.  With CUB-200, for
example, ``training`` selects from classes 1--100 and ``testing`` selects from
classes 101--200.  Class labels are re-indexed later by the dataset loader in
exactly the same way as a normal run.
"""


def select_quick_experiment_classes(opt, classes, split_name):
    """Return the requested leading class prefix, or all classes when off."""
    classes = list(classes)
    if not int(getattr(opt, "subset_experiment_enabled", 0)):
        return classes

    if split_name == "training":
        count = int(getattr(opt, "quick_train_class_count", 0))
    elif split_name in ("testing", "query_gallery"):
        count = int(getattr(opt, "quick_test_class_count", 0))
    else:
        raise ValueError(
            "Unknown quick-experiment split: {}".format(split_name))

    if count <= 0 or count > len(classes):
        raise ValueError(
            "quick experiment {} class count must be in [1, {}], got {}."
            .format(split_name, len(classes), count))

    selected = classes[:count]
    print(
        "Quick experiment class slice: {} first {}/{} classes"
        .format(split_name, count, len(classes)))
    return selected
