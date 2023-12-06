# Copyright (c) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from textwrap import dedent


def get_macs(model, input_, method="fvcore", verbose=False):
    if method == "fvcore":
        from fvcore.nn import FlopCountAnalysis

        flops = FlopCountAnalysis(model, input_)
        if verbose:
            print(
                dedent(
                    f"""\
            fvcore
            ======
            flops.total():                  {flops.total()}
            flops.by_operator():            {flops.by_operator()}
            flops.by_module():              {flops.by_module()}
            flops.by_module_and_operator(): {flops.by_module_and_operator()}
            """
                )
            )
        macs = flops.total()
    elif method == "thop":
        from thop import profile

        macs, total_params = profile(model, (input_,), verbose=verbose)
        if verbose:
            print(
                dedent(
                    f"""\
            thop
            ====
            total_ops:                      {macs}
            total_params:                   {total_params}
            """
                )
            )
    elif method == "ptflops":
        # This one seems very inaccurate at times
        from ptflops import get_model_complexity_info

        macs, params = get_model_complexity_info(
            model, tuple(input_.size()[1:]), as_strings=False, print_per_layer_stat=verbose, verbose=verbose
        )
        if verbose:
            print(
                dedent(
                    f"""\
            ptflops
            ====
            macs:                           {macs}
            flops (est.):                   {2 * macs}
            params:                         {params}
            """
                )
            )
    else:
        raise NotImplementedError(f'MAC-counting method "{method}"')
    return int(macs)


def benchmark():
    import torch

    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
    model.eval()
    input_ = torch.rand(1, 3, 224, 224)

    get_macs(model, input_, method="fvcore", verbose=True)
    get_macs(model, input_, method="thop", verbose=True)
    get_macs(model, input_, method="ptflops", verbose=True)


if __name__ == "__main__":
    benchmark()
