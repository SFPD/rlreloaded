#!/usr/bin/env python
from control4.bench.bench import ScriptRun,prepare_dir_for_experiment,assert_script_runs_different,ordered_load
from control4.cloud.cloud_interface import load_cloud_config,create_cloud
from control4.misc.collection_utils import chunkify, dict_update, filter_dict
from control4.misc.console_utils import yes_or_no, maybe_call_and_print
from control4.misc.randomness import random_string
import threading, subprocess, time, os.path as osp
from fnmatch import fnmatch
from collections import OrderedDict
from control4.algs.alg_params import string2dict
import argparse

def filter_dict_by_desc(d,desc):
    if "," in desc:
        scriptnames = desc.split(",")
        filtfn = lambda s: s in scriptnames
    elif "*" in desc:
        filtfn = lambda s: fnmatch(s, desc)
    else:
        filtfn = lambda s: s==desc
    return filter_dict(d, lambda key,_: filtfn(key))

def oneof(*args):
    s = 0
    for arg in args: s += bool(arg)
    return s==1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_file",type=argparse.FileType("r"))
    parser.add_argument("out_root")
    parser.add_argument("--script_include")
    parser.add_argument("--cfg_include")
    parser.add_argument("--test_include")

    parser.add_argument("--gce",action="store_true")
    parser.add_argument("--one_script_per_machine",action="store_true")
    parser.add_argument("--one_run_per_machine",action="store_true")
    parser.add_argument("--scripts_per_machine",type=int)
    parser.add_argument("--keep_instance",action="store_true")
    parser.add_argument("--instance_prefix")
    parser.add_argument("--test",action="store_true")
    parser.add_argument("--n_runs",type=int)
    parser.add_argument("--alter_default_settings",type=str)
    parser.add_argument("--cfg_name_prefix",type=str,default="")
    parser.add_argument("--cont",action="store_true")
    parser.add_argument("--parallel_local",action="store_true")
    parser.add_argument("--pipe_to_logfile",choices=["off","all","stdout"],default="all")
    parser.add_argument("--start_from")
    parser.add_argument("--start_run",type=int,default=0)


    parser.add_argument("--dry",action="store_true")
    args = parser.parse_args()
    if args.gce: 
        assert oneof(args.one_script_per_machine,args.one_run_per_machine,args.scripts_per_machine)
        assert not args.test
        output = subprocess.check_output("cd $CTRL_ROOT && git cherry -v",shell=True)
        if len(output.strip()) > 0:
            if not yes_or_no("Running on gce but you have unpushed commits. Continue?"):
                print "Exiting"
                exit(1)

    expt_info = ordered_load(args.yaml_file)
    n_runs = args.n_runs or expt_info.get("n_runs",1)
    default_settings = expt_info.get("default_settings",{})
    if args.alter_default_settings: default_settings.update(string2dict(args.alter_default_settings))
    if args.test: 
        default_settings["test_mode"]=1
        args.pipe_to_logfile = "off"
        n_runs = 1

    if 'cfg_name' in default_settings: default_settings['cfg_name'] = str(default_settings['cfg_name']) # Deal with stupid bug where commit name gets turned into int
    out_root = args.out_root
    if out_root.startswith("/"):
        if args.gce: raise RuntimeError("Don't specify an absolute path for output when running jobs on GCE")
    else:
        assert not "results" in out_root
        out_root = osp.join("$CTRL_DATA","results",out_root)

    if not (args.dry or args.test): prepare_dir_for_experiment(out_root, args.cont)

    assert "scripts" in expt_info or ("tests" in expt_info and "cfgs" in expt_info)

    if "scripts" in expt_info:
        script_dict = expt_info["scripts"]
        if args.script_include is not None:
            script_dict = filter_dict_by_desc(script_dict,args.script_include)
        assert args.test_include is None and args.cfg_include is None,"can't use {cfg/test}_include when yaml file has scripts:"
    else:
        test_dict = expt_info["tests"]
        if args.test_include: test_dict = filter_dict_by_desc(test_dict,args.test_include)
        cfg_dict = expt_info["cfgs"]
        if args.cfg_include: cfg_dict = filter_dict_by_desc(cfg_dict,args.cfg_include)
        script_dict = {}
        for (testname,testinfo) in test_dict.items():
            for (cfgname,cfginfo) in cfg_dict.items():
                scriptinfo = {}
                scriptinfo["test_name"] = testname
                scriptinfo["cfg_name"] = args.cfg_name_prefix + cfgname
                scriptinfo.update(testinfo)
                scriptinfo.update(cfginfo)
                scriptname = cfgname + "-" + testname
                script_dict[scriptname] = scriptinfo
        assert args.script_include is None,"For this type of yaml file, you should use cfg_include/test_include"


    if args.start_from is not None:
        pairs = []
        gotit = False
        for (k,v) in script_dict.items():
            if k==args.start_from:
                gotit=True
            if gotit:
                pairs.append((k,v))
            else:
                print "skipping",k
        script_dict = OrderedDict(pairs)
    assert len(script_dict) > 0

    all_srs = []
    for i_run in xrange(args.start_run, n_runs):
        for (script_name,script_info) in script_dict.items():
            script_info = dict_update(default_settings, script_info)
            sr = ScriptRun(script_info,script_name,i_run, out_root)
            all_srs.append(sr)

    assert_script_runs_different(all_srs)

    if args.gce:
        if args.one_script_per_machine: scripts_per_machine = 1
        elif args.one_run_per_machine: scripts_per_machine = len(script_dict)
        else: scripts_per_machine = args.scripts_per_machine
        instance_prefix = args.instance_prefix or random_string(4)
        threads = []
        for (i_inst,srs) in enumerate(chunkify(all_srs, scripts_per_machine)):
            cmds = [sr.get_cmd(pipe_to_logfile=args.pipe_to_logfile) for sr in srs]
            instance_name = "%s-%.4i"%(instance_prefix,i_inst)

            cloud_config = load_cloud_config(provider='gce')
            cloud = create_cloud(cloud_config)
            th = threading.Thread(target = cloud.run_commands_on_fresh_instance,
                args = (cmds,instance_name), 
                kwargs=dict(dry=args.dry, keep_instance=args.keep_instance))
            th.start()
            time.sleep(.1)
            threads.append(th)
        print "Starting %i GCE instances"%len(threads)
        for th in threads:
            th.join()
        print "Done starting %i GCE instances"%len(threads)


    else:
        for sr in all_srs:
            maybe_call_and_print(sr.get_cmd(pipe_to_logfile=args.pipe_to_logfile),args.dry)
                


if __name__ == "__main__":
    main()