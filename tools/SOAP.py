import numpy as np
import pysobatools.sobaeval as SOAPEval
import pysobatools.cocoeval as Eval
from pysobatools.soba import SOBA
import json
import argparse




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
    parser.add_argument("--path", required=True)
    parser.add_argument("--input-name", required=True)
    args = parser.parse_args()
    soba = SOBA(args.path)
    results =  json.load(open('{}/inference/soba_instances_results.json'.format(args.input_name)))
    association = json.load(open('{}/inference/soba_association_results.json'.format(args.input_name)))

    instance_soba = soba.loadRes(results)
    association_soba = soba.loadRes_asso(association)

    sobaeval= SOAPEval.SOAPeval(soba,instance_soba,association_soba)
    print('segmentaion:')

    sobaeval.evaluate_asso()

    sobaeval.accumulate()
    sobaeval.summarize()
    print('bbox:')
    sobaeval= SOAPEval.SOAPeval(soba,instance_soba,association_soba)
    sobaeval.params.iouType = 'bbox'
    sobaeval.evaluate_asso()

    sobaeval.accumulate()
    sobaeval.summarize()

    print("--------------")
    sobaeval= Eval.COCOeval(soba,association_soba)
    sobaeval.evaluate_asso()
    sobaeval.accumulate()
    sobaeval.summarize()

    sobaeval= Eval.COCOeval(soba,association_soba)
    sobaeval.params.iouType="bbox"
    sobaeval.evaluate_asso()
    sobaeval.accumulate()
    sobaeval.summarize()