from source.descriptors import DESC_CLS, DESC_DST
from source.experiments.experiments import Experiment

import argparse


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('-d', '--dataset', nargs='+', default=['acm','20ng','dblp','wos11967','books','agnews','sogou'],
						help='Datasets to run.', choices=DESC_DST.keys())

	parser.add_argument('-m', '--method', nargs='+', default=['etc-base', 'bert'],
						help='Methods to run.', choices=DESC_CLS.keys())

	parser.add_argument('-f', '--force', nargs='+', type=int, default=None, help='Define the fold_idx to force run.')

	args = parser.parse_args()
	print(args)

	datasets = [ DESC_DST[d] for d in args.dataset ]
	trainers = [ DESC_CLS[m] for m in args.method ]

	exp = Experiment( dst_descs=datasets, trnr_descs=trainers, force=args.force, output_path='~/.etc_alt/')
	exp.run()
