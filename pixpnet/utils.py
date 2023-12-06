# Copyright (c) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: CC-BY-SA-4.0

"""
============================================================================
Attribution for the function `is_pareto_efficient(...)`
(https://stackoverflow.blog/2009/06/25/attribution-required/):
============================================================================
Question and answer from StackOverflow:
    https://stackoverflow.com/q/32791911
    https://stackoverflow.com/a/40239615
Thank you to the author of the question:
    Lucien S. (https://stackoverflow.com/users/1208142/lucien-s)
Thank you to the author of the answer:
    Peter (https://stackoverflow.com/users/851699/peter)
Thank you to the authors of the other answers:
    hilberts_drinking_problem (https://stackoverflow.com/users/4585963/hilberts-drinking-problem)
    elzurdo (https://stackoverflow.com/users/6763056/elzurdo)
    jmmcd (https://stackoverflow.com/users/86465/jmmcd)
    Ragheb (https://stackoverflow.com/users/5004778/ragheb)
    Jean Claude (https://stackoverflow.com/users/14801243/jean-claude)
    fabi lauchi (https://stackoverflow.com/users/8775937/fabi-lauchi)
============================================================================
"""
import argparse
import importlib.util
import inspect
import logging
import os
import os.path as osp
import warnings
from collections import OrderedDict
from datetime import datetime
from math import log10
from pprint import pformat
from typing import IO, Any, AnyStr, Optional, Sequence, Union

import configargparse
import numpy as np
import yaml

try:
    from yaml import CBaseDumper as yaml_BaseDumper
    from yaml import CBaseLoader as yaml_BaseLoader
    from yaml import CDumper as yaml_Dumper
    from yaml import CLoader as yaml_Loader
    from yaml import CSafeDumper as yaml_SafeDumper
    from yaml import CSafeLoader as yaml_SafeLoader
except ImportError:
    from yaml import BaseDumper as yaml_BaseDumper
    from yaml import BaseLoader as yaml_BaseLoader
    from yaml import Dumper as yaml_Dumper
    from yaml import Loader as yaml_Loader
    from yaml import SafeDumper as yaml_SafeDumper
    from yaml import SafeLoader as yaml_SafeLoader

    warnings.warn("Could not load CSafeLoader.")


# noinspection PyPep8Naming
def yaml_load(
    stream: Union[IO, AnyStr],
    Loader: Optional[yaml_BaseLoader] = None,
    safe: Optional[bool] = None,
    load_all: bool = False,
) -> Any:
    """
    A wrapper function around PyYAML loading functions. By default, the
    `yaml.SafeLoader` loader is used. If the C bindings for YAML functions
    are available, then they are used instead of the Python implementations
    (e.g., `yaml.CSafeLoader` is preferred over `yaml.SafeLoader`).

    :param stream: a file-like object supporting `read()` or any string
        containing the YAML data to be loaded
    :param Loader: the YAML loader class to use
    :param safe: whether to use the `SafeLoader` or `Loader` loader. If a
        loader is not provided, then `safe` defaults to `True`. Do not
        provide this argument if you explicitly specify the `Loader`
    :param load_all: whether to load all YAML documents in `stream` or just
        the first one
    :return: the loaded YAML as a Python object(s)
    :raises ValueError: if both `Loader` and `safe` are specified
    """
    if Loader is None:
        if safe is None:
            safe = True
    elif safe is not None:
        raise ValueError(f"If `Loader` is provided, then `safe` must be " f"`None`. However, `safe={safe}` was passed.")
    load_func = yaml.load_all if load_all else yaml.load
    loader = yaml_SafeLoader if safe else (Loader or yaml_Loader)
    return load_func(stream, loader)


# noinspection PyPep8Naming
def yaml_dump(
    data: Any,
    stream: Optional[Union[IO, AnyStr]] = None,
    Dumper: Optional[yaml_BaseDumper] = None,
    safe: Optional[bool] = None,
    dump_all: bool = False,
    **kwargs,
) -> Optional[AnyStr]:
    """
    A wrapper function around PyYAML dumping functions. By default, the
    `yaml.SafeDumper` dumper is used. If the C bindings for YAML functions
    are available, then they are used instead of the Python implementations
    (e.g., `yaml.CSafeDumper` is preferred over `yaml.SafeDumper`).

    :param data: a Python object(s) to dump
    :param stream: a file-like object supporting `read()` or any string
        containing the YAML data to be loaded
    :param Dumper: the YAML dumper class to use
    :param safe: whether to use the `SafeLoader` or `Loader` loader. If a
        loader is not provided, then `safe` defaults to `True`. Do not
        provide this argument if you explicitly specify the `Loader`
    :param dump_all: whether to dump a sequence of Python objects or just a
        single object
    :return: the dumped YAML as `str` or `bytes` depending on the optional
        `encoding` kwarg
    :raises ValueError: if both `Loader` and `safe` are specified
    """
    if Dumper is None:
        if safe is None:
            safe = True
    elif safe is not None:
        raise ValueError(f"If `Dumper` is provided, then `safe` must be " f"`None`. However, `safe={safe}` was passed.")
    dump_func = yaml.dump_all if dump_all else yaml.dump
    dumper = yaml_SafeDumper if safe else (Dumper or yaml_Dumper)
    return dump_func(data, stream=stream, Dumper=dumper, **kwargs)


def get_logger(name):
    logging.basicConfig(
        format="%(asctime)s[%(process)d][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    logger = logging.getLogger(name)
    logger.setLevel(os.environ.get("PIXPNET_LOG_LEVEL", "INFO"))
    return logger


logged = set()


def log_once(printer, msg):
    if msg in logged:
        return
    logged.add(msg)
    printer(msg)


def ns_to_nested_ns(namespace):
    if not isinstance(namespace, dict):
        namespace = vars(namespace)
    ns_nested = {}
    for k, v in namespace.items():
        ks = k.split(".", 1)
        if len(ks) == 1:
            ns_nested[k] = v
        else:
            ka, kb = ks
            ns_nested_a = ns_to_nested_ns({kb: v})
            if ka in ns_nested:
                setattr(ns_nested[ka], kb, getattr(ns_nested_a, kb))
            else:
                ns_nested[ka] = ns_nested_a
    return configargparse.Namespace(**ns_nested)


def nested_ns_to_nested_dict(namespace):
    d = {
        k: nested_ns_to_nested_dict(v) if isinstance(v, configargparse.Namespace) else v
        for k, v in vars(namespace).items()
    }
    return d


def flatten_nested_dict(d, prefix=""):
    d_flat = {}
    for k, v in d.items():
        k = ((prefix + ".") if prefix else "") + k
        if isinstance(v, dict):
            d_flat.update(flatten_nested_dict(v, prefix=k))
        else:
            d_flat[k] = v
    return d_flat


def get_all_func_args(func):
    try:
        argspec = inspect.getfullargspec(func)
    except TypeError:  # unsupported callable
        return []
    is_cls = inspect.isclass(func)
    all_args = argspec.args + argspec.kwonlyargs
    if is_cls:
        all_args = all_args[1:]  # drop self
        if argspec.varargs or argspec.varkw:
            for super_ in func.__mro__[1:]:  # exclude func cls
                all_args.extend(get_all_func_args(super_))
            all_args = [*{*all_args}]
    return all_args


def intersect_func_and_kwargs(func, kwargs, exclude_func_args=None, exclude_kwargs=None, return_invalid=True):
    func_args = {*get_all_func_args(func)} - (set() if exclude_func_args is None else {*exclude_func_args})
    if isinstance(kwargs, argparse.Namespace):
        kwargs = vars(kwargs)
    kwargs_keys = {*kwargs.keys()} - (set() if exclude_kwargs is None else {*exclude_kwargs})

    intersecting_keys = kwargs_keys & func_args
    intersected_dict = {k: kwargs[k] for k in intersecting_keys}
    if return_invalid:
        return intersected_dict, kwargs_keys - func_args
    return intersected_dict


def now_str():
    return datetime.now().isoformat(timespec="seconds").replace(":", "_")


def num_cpus():
    if "SLURM_CPUS_PER_TASK" in os.environ:
        # Slurm env
        try:
            return int(os.environ["SLURM_CPUS_PER_TASK"])
        except ValueError:
            raise RuntimeError(
                f"Detected SLURM environment but "
                f"$SLURM_CPUS_PER_TASK is not an int: "
                f'{os.environ["SLURM_CPUS_PER_TASK"]}'
            )
    elif "JOB_ID" in os.environ:
        # assume SGE environment
        base_err = (
            "Inferred that you are in an SGE environment (because "
            f'$JOB_ID is set as {os.environ["JOB_ID"]}) but $NSLOTS '
            f"is not "
        )
        try:
            return int(os.environ["NSLOTS"])
        except KeyError:
            raise RuntimeError(base_err + "set!")
        except ValueError:
            raise RuntimeError(base_err + f'an int ({os.environ["NSLOTS"]})!')
    else:
        # assume no scheduler (resource allocation)
        return os.cpu_count()


class NestedYAMLConfigFileParser(configargparse.YAMLConfigFileParser):
    def parse(self, stream, as_str=True, ignore_none=True):
        # see ConfigFileParser.parse docstring
        yaml = self._load_yaml()
        if isinstance(yaml, tuple):
            yaml = yaml[0]

        try:
            parsed_obj = yaml.safe_load(stream)
        except Exception as e:
            raise configargparse.ConfigFileParserException("Couldn't parse config file: %s" % e)

        if not isinstance(parsed_obj, dict):
            raise configargparse.ConfigFileParserException(
                "The config file doesn't appear to "
                "contain 'key: value' pairs (aka. a YAML mapping). "
                "yaml.load('%s') returned type '%s' instead of 'dict'."
                % (getattr(stream, "name", "stream"), type(parsed_obj).__name__)
            )

        if not ignore_none:
            assert not as_str
        result = self._parse_level(parsed_obj, as_str=as_str, ignore_none=ignore_none)
        return result

    def _parse_level(self, parsed_obj, as_str, ignore_none):
        result = OrderedDict()
        for key, value in parsed_obj.items():
            if isinstance(value, list):
                result[key] = value
            elif isinstance(value, dict):
                result_nested = self._parse_level(value, as_str, ignore_none)
                for key_nested, value_nested in result_nested.items():
                    result[key + "." + key_nested] = value_nested
            elif ignore_none and value is None:
                pass
            else:
                result[key] = str(value) if as_str else value

        return result


def load_module_copy(module_name):
    # load a copy of a module to avoid any global consequences
    module_spec = importlib.util.find_spec(module_name)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


# noinspection PyUnresolvedReferences
def make_base_parser(description, sub_log_dir, proj_path):
    parser = configargparse.ArgParser(
        description=description,
        config_file_parser_class=NestedYAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add("-c", "--config", is_config_file=True, help="config file path")

    parser.add("--seed", type=int, help="The random seed")
    parser.add(
        "--gpus",
        type=int,
        default=None,
        help="The number of GPUs to use. If None, 1 GPU is used if " "available, otherwise code will be run CPU-only.",
    )
    parser.add("--log-dir", default=osp.join(proj_path, "logs", sub_log_dir), help="The logging directory")
    parser.add("--run-id", default=None, help="Identifier for this run that suffixes the subdirectory " "in --log-dir")
    parser.add("--profile", action="store_true", help="profile code")
    parser.add(
        "--profiler", default="simple", choices=["simple", "advanced", "pytorch", "xla"], help="the profiler name"
    )
    parser.add("--debug", action="store_true", help="debug flag")

    dataset = parser.add_argument_group("dataset")
    dataset.add("--dataset.name", default="MNIST", type=str, help="The dataset name")
    dataset.add("--dataset.val-size", default=0.2, type=float, help="The size of the validation split")
    dataset.add(
        "--dataset.augment-factor", default=1, type=int, help="The number of augmented images per training image"
    )
    dataset.add("--dataset.root", help="The dataset root directory")
    dataset.add(
        "--dataset.crop-to-bbox", action="store_true", help="Crop applicable datasets to bounding box annotations"
    )
    # shhh I am internal ignore me
    dataset.add("--dataset.needs_unaugmented", help=argparse.SUPPRESS, type=bool, default=False)

    train = parser.add_argument_group("train")
    train.add("--train.epochs", default=50, type=int, help="Number of epochs to train for")
    train.add("--train.batch-size", type=int, default=128, help="Training batch size")
    train.add(
        "--train.accumulate-grad-batches",
        type=int,
        default=None,
        help="Number of batches to accumulate gradients before backprop. "
        "In effect, increases the batch size by this factor.",
    )
    train.add(
        "--train.checkpoint",
        action="store_true",
        default=False,
        help="Checkpoint the model and save/restore the best-performing " "(on the validation set) for testing",
    )
    train.add("--train.no-checkpoint", action="store_false", dest="train.checkpoint")
    train.add("--train.gradient-clip-norm", type=float, default=None, help="Gradient clip norm value")
    train.add(
        "--train.hparam-tune",
        action="store_true",
        help="Tune hyperparameters before training (see "
        "`pytorch_lightning.Trainer.tune` documentation for "
        "details).",
    )
    train.add(
        "--train.stochastic-weight-averaging",
        action="store_true",
        help="Perform stochastic weight averaging during training.",
    )
    train.add(
        "--train.val-every-n-epoch",
        type=int,
        help="Evaluate validation set every n epochs. By default " "select a value that does not evaluate too often",
    )

    test = parser.add_argument_group("test")
    test.add(
        "--test.batch-size",
        type=int,
        default=None,
        help="Testing batch size. Double that of training batch size by " "default (if None)",
    )

    optimizer = parser.add_argument_group("optimizer")
    optimizer.add("--optimizer.name", default="sgd", help="optimizer name")
    optimizer.add("--optimizer.lr", type=float, default=0.1, help="optimizer learning rate")
    optimizer.add(
        "--optimizer.lr-scheduler",
        default="cosine",
        choices=["cosine", "step", "multistep"],
        help="The name of the learning rate scheduler",
    )
    optimizer.add("--optimizer.lr-factor", default=0.1, type=float, help="Learning rate decay factor (step scheduler)")
    optimizer.add(
        "--optimizer.lr-schedule",
        nargs="+",
        default=[100, 150],
        type=int,
        help="Milestones when the learning rate is dropped " "by lr-factor (step scheduler)",
    )
    optimizer.add(
        "--optimizer.weight-decay", type=float, default=2e-4, help="Weight decay of the parameters (L2 penalty)"
    )
    optimizer.add("--optimizer.momentum", type=float, default=0.9, help="Momentum of the optimizer (if applicable)")
    optimizer.add(
        "--optimizer.warmup-period",
        type=int,
        default=5,
        help="The number of epochs to exponentially warm up the " "learning rate",
    )

    return parser, dataset, train, test, optimizer


def parse_args(parser, logger):
    config = parser.parse_args()
    config_nested = ns_to_nested_ns(config)
    logger.info(parser.format_values())
    logger.info(pformat(nested_ns_to_nested_dict(config_nested)))
    return config_nested


def parse_config_file(filename):
    with open(filename, "r") as fp:
        parsed = NestedYAMLConfigFileParser().parse(fp, as_str=False, ignore_none=False)

    config = argparse.Namespace()
    for k, v in parsed.items():
        setattr(config, k, v)

    return ns_to_nested_ns(config)


def is_pareto_efficient(
    costs: np.ndarray,
    return_mask: bool = True,
    maximize: Union[bool, Sequence[bool]] = True,
) -> np.ndarray:
    """
    ============================================================================
    Attribution (https://stackoverflow.blog/2009/06/25/attribution-required/):
    ============================================================================
    Question and answer from StackOverflow:
        https://stackoverflow.com/q/32791911
        https://stackoverflow.com/a/40239615
    Thank you to the author of the question:
        Lucien S. (https://stackoverflow.com/users/1208142/lucien-s)
    Thank you to the author of the answer:
        Peter (https://stackoverflow.com/users/851699/peter)
    Thank you to the authors of the other answers:
        hilberts_drinking_problem (https://stackoverflow.com/users/4585963/hilberts-drinking-problem)
        elzurdo (https://stackoverflow.com/users/6763056/elzurdo)
        jmmcd (https://stackoverflow.com/users/86465/jmmcd)
        Ragheb (https://stackoverflow.com/users/5004778/ragheb)
        Jean Claude (https://stackoverflow.com/users/14801243/jean-claude)
        fabi lauchi (https://stackoverflow.com/users/8775937/fabi-lauchi)
    ============================================================================

    Find the pareto-efficient points

    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :param maximize: True if maximizing
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    assert costs.ndim == 2
    costs_ = np.copy(costs)
    costs_[np.isnan(costs_)] = np.inf
    if isinstance(maximize, bool):
        maximize = [maximize] * costs_.shape[1]
    else:
        assert len(maximize) == costs_.shape[1]
    for i, maximize_col in enumerate(maximize):
        if maximize_col:
            costs_[:, i] = -costs_[:, i]
    is_efficient, n_points = np.arange(costs_.shape[0]), costs_.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs_):
        nondominated_point_mask = np.any(costs_ < costs_[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        # Remove dominated points
        is_efficient = is_efficient[nondominated_point_mask]
        costs_ = costs_[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    return is_efficient  # else


def nondominating(
    costs: np.ndarray,
    maximize: Union[bool, Sequence[bool]] = True,
) -> np.ndarray:
    mask = is_pareto_efficient(costs, maximize=maximize, return_mask=True)
    return costs[mask]


def pretty_si_units(value, places=2):
    if value == 0:
        return "0"
    prefixes = ["y", "z", "a", "f", "p", "n", "μ", "m", "", "k", "M", "G", "T", "P", "E", "Z", "Y"]
    decades_div_3 = max(min(int(log10(value) // 3), 8), -8)
    prefix = prefixes[decades_div_3 + 8]
    space = " " if prefix else ""
    value_si_raw = value / 1000**decades_div_3
    value_si = round(value_si_raw, places)
    if value_si == 0:
        value_si = round(value_si_raw, abs(int(log10(value_si_raw))))
    if value_si.is_integer():
        value_si = int(value_si)
    return f"{value_si}{space}{prefix}"
