#!/usr/bin/env python
"""
Run the path-based stochastic policy iteration algorithm

"""


import argparse
from control4.algs.trpo_single_path import single_path_spi, configure_test_mode
from control4.algs.alg_params import *
from control4.config import setup_logging,print_theano_config
from control4.misc.console_utils import dict_as_table
import numpy as np,os,sys



def main():

    setup_logging()
    np.set_printoptions(precision=3)
    parser = argparse.ArgumentParser(formatter_class=lambda prog : argparse.ArgumentDefaultsHelpFormatter(prog,max_help_position=50))
    param_list = [GeneralScriptParams,ProfileParams,DiagnosticParams,
                    MDPParams,SinglePathEstimationParams,
                    SPIParams,PolicyOptParams,VFOptParams]
    for param in param_list:
        param.add_to_parser(parser)
    args = parser.parse_args()
    params = args.__dict__
    params.update(string2dict(params["misc_kws"]))
    validate_and_filter_args(param_list, args)
    if params['test_mode']: configure_test_mode(params)
    print_theano_config()
    print dict_as_table(params)
    if sys.platform == 'darwin' and args.par_mode != "off":
        assert os.getenv("VECLIB_MAXIMUM_THREADS")=="1"

    if args.alg == "single_path":
        single_path_spi(params)
    else:
        raise NotImplementedError




if __name__ == "__main__":
    main()
