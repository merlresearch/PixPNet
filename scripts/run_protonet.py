# Copyright (c) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os.path as osp

proj_path = osp.dirname(osp.dirname(osp.realpath(__file__)))

try:
    import pixpnet
except ImportError:
    import sys

    sys.path.append(proj_path)

    import pixpnet  # noqa: F401


def main():
    from pixpnet.utils import make_base_parser, parse_args

    parser, dataset, train, test, optimizer = make_base_parser("Run protonet pipeline", "protonet", proj_path)

    parser.add(
        "--tb-logging", action="store_true", help="Verbose logging of activations and gradients for " "TensorBoard"
    )

    model = parser.add_argument_group("model")
    model.add("--model.name", required=True, choices=["protonet"], help="name of the model")
    model.add("--model.feature-extractor", required=True, help="name of the feature extractor")
    model.add("--model.feature-layer", default=None, help="name of the feature extractor layer to use")
    model.add(
        "--model.pretrained",
        action="store_true",
        default=True,
        help="whether the feature extractor should be pretrained on " "ImageNet",
    )
    model.add("--model.no-pretrained", action="store_false", dest="model.pretrained")
    model.add("--model.num-prototypes", type=int, default=500, help="the number of prototypes in the protonet")
    model.add("--model.prototype-dim", type=int, default=512, help="the dimensionality of the prototypes")
    model.add(
        "--model.prototype-kernel-size",
        type=int,
        default=1,
        help="the kernel size of the prototypes (size of patches in " "latent space)",
    )
    model.add(
        "--model.init-weights",
        action="store_true",
        default=True,
        help="whether to initialize the weights of the protonet",
    )
    model.add("--model.no-init-weights", action="store_false", dest="model.init_weights")
    model.add(
        "--model.prototype-activation",
        choices=["log", "linear"],
        default="log",
        help="the prototype activation function",
    )
    model.add(
        "--model.add-on-layers-type",
        default="regular",
        choices=["regular", "bottleneck"],
        help="the type of add-on layers after the feature extractor to " "change channel size",
    )
    model.add(
        "--model.epsilon",
        type=float,
        default=1e-6,
        help="the epsilon value in the divisor of the log prototype " "activation function",
    )
    model.add(
        "--model.learn-prototypes",
        action="store_true",
        default=True,
        help="whether to learn prototypes by gradient descent",
    )
    model.add("--model.no-learn-prototypes", action="store_false", dest="model.learn_prototypes")
    model.add(
        "--model.incorrect-strength",
        type=float,
        default=-0.5,
        help="the initialization strength of incorrect connections in " "the readout weights (class-wise connections)",
    )
    model.add(
        "--model.correct-strength",
        type=float,
        default=1.0,
        help="the initialization strength of correct connections in " "the readout weights (class-wise connections)",
    )
    model.add(
        "--model.class-specific",
        action="store_true",
        default=True,
        help="whether model and its training should have class-specific " "prototypes",
    )
    model.add("--model.no-class-specific", action="store_false", dest="model.class_specific")
    model.add(
        "--model.readout-type", default="linear", choices=["linear", "sparse", "proto"], help="The readout layer type"
    )
    model.add(
        "--model.distance",
        default="l2",
        choices=["l2", "cosine"],
        help="The distance metric used between latent patches and " "prototypes",
    )

    train.add(
        "--train.push-prototypes",
        action="store_true",
        default=True,
        help="replace prototypes every --train.push-every epochs",
    )
    train.add("--train.no-push-prototypes", action="store_false", dest="train.push_prototypes")
    train.add("--train.push-every", type=int, default=10, help="replace prototypes every X epochs if pushing enabled")
    train.add(
        "--train.readout-push-epochs",
        type=int,
        default=20,
        help="if replacing prototypes, train readout layer for this many " "epochs",
    )
    train.add(
        "--train.push-duplicate-filter",
        default="sample",
        choices=["sample", "patch", "none"],
        help="if replacing prototypes, whether and how to prevent " "duplicate prototype assignment",
    )

    optimizer.add(
        "--optimizer.fine-tune-lr",
        type=float,
        default=1e-4,
        help="fine-tuning learning rate of the feature extractor " "(if pretrained)",
    )
    optimizer.add("--optimizer.readout-lr", type=float, default=1e-4, help="readout/classification layer learning rate")

    loss = parser.add_argument_group("loss")
    loss.add("--loss.xent", type=float, default=1.0, help="loss term coefficient for xent loss")
    loss.add("--loss.cluster", type=float, default=0.8, help="loss term coefficient for cluster loss")
    loss.add("--loss.separation", type=float, default=0.08, help="loss term coefficient for separation loss")
    loss.add("--loss.l1", type=float, default=1e-4, help="loss term coefficient for l1 loss")

    from pixpnet.utils import get_logger

    logger = get_logger("run_protonet")
    config_nested = parse_args(parser, logger)

    # set needs_unaugmented appropriately
    config_nested.dataset.needs_unaugmented = config_nested.train.push_prototypes

    # now that args have been parsed load the heavier modules
    # > TensorFlow needs to be imported before PyTorch Lightning!
    # https://github.com/pytorch/pytorch/issues/81140#issuecomment-1230524321
    try:
        import tensorflow as tf  # noqa
    except ImportError:
        pass
    from pixpnet.pipeline import run
    from pixpnet.protonets.lit_model import ProtoFitLoop, ProtoLitModel

    run(config_nested, ProtoLitModel, ProtoFitLoop)


if __name__ == "__main__":
    main()
