import json
import os
import argparse
import pickle
import re
from transformers import *
import torch
from scipy import stats
from collections import defaultdict, Counter
import numpy as np
import csv
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from read_data import *
import seaborn as sns
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--t_test", action='store_true')
parser.add_argument("--dataset_statistics", action='store_true')
parser.add_argument("--analysis", action='store_true')
parser.add_argument("--multik", action='store_true')
parser.add_argument("--length_plot", action='store_true')
parser.add_argument("--beauty_plot", action='store_true')
parser.add_argument("--nrb", action='store_true')
parser.add_argument("--LMBFF", action='store_true')
parser.add_argument("--example_id", default=0, type=int)
parser.add_argument("--max_seq_length", default=256, type=int)
parser.add_argument("--model_name", default='bert-base-cased', type=str)
parser.add_argument("--visualize_per_sample", action='store_true', help='whether to visualize per sample')
parser.add_argument("--visualize_per_class", action='store_true', help='whether to visualize per class')
parser.add_argument("--task", type=str, default='ner')
parser.add_argument("--dataset", type=str, default='conll2003')
parser.add_argument("--tagging", type=str, default='IO')
parser.add_argument("--k", type=str, default='5shots')
parser.add_argument("--mode", nargs="+", type=str, default=("no", "standard", "standard_wrong", "standard_no_l", "random_totally", "random_support"))
parser.add_argument("--output_file", type=str, default="try")
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--metric", nargs="+", default=("f1", "precision", "recall"))
parser.add_argument("--text", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--no_errbar", action="store_true")
parser.add_argument("--id", nargs="+", default=(1, 2, 3, 4, 5))
parser.add_argument("--seed", nargs="+", default=(11, 42, 55))
parser.add_argument("--data_dir", default=None)
args = parser.parse_args()
data_dir = os.path.join('out', args.task + '-' + args.k, args.dataset + '-' + args.tagging)
if args.data_dir is not None:
	data_dir = args.data_dir
format_mode = {'no': 'NO', 'standard': 'ST', 'standard_wrong': 'SW', 'standard_no_l': 'SN', 'random_totally': 'TR', 'random_support': 'SR'}
format_dataset = {'conll2003': 'CoNLL03', 'ontonotes': 'Ontonotes 5.0', 'conll2000': 'CoNLL00'}
format_class = {'LOCATION': 'LOC', 'ORGANIZATION': 'ORG', 'PERSON': 'PER', 'O': 'O'}


def t_test(a1, a2):
	# Paired Student's t-test: Calculate the T-test on TWO RELATED samples of scores, a and b.
	# for one sided test we multiply p-value by half
	t_results = stats.ttest_rel(a1, a2)
	# correct for one sided test
	pval = float(t_results[1]) / 2
	if float(pval) <= 0.05:
		print("\nTest result is significant with p-value: {}".format(pval))
	else:
		print("\nTest result is not significant with p-value: {}".format(pval))


def get_sorted_labels(task, dataset):
	if task == 'ner':
		if dataset == 'conll2003':
			return ["MISC", "PER", "ORG", "LOC"]
		elif dataset == 'ontonotes':
			return ["LANGUAGE", "LAW", "PRODUCT", "EVENT", "FAC", "WORK_OF_ART", "LOC", "NORP", "GPE", "PERSON", "ORG"]
		elif dataset in ('NRB', 'WTS'):
			return ["ORGANIZATION", "LOCATION", "PERSON", ]
		else:
			raise ValueError("{} dataset is not supported, only conll2003, ontonotes are available".format(dataset))
	else:
		if dataset == 'conll2000':
			return ["NP", "VP", "PP", "ADVP", "SBAR", "ADJP"]


def dataset_statistics(file_path):
	label_map = get_sorted_labels(args.task, args.dataset)
	tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
	results = {}
	with open(file_path, encoding="utf-8") as f:
		words = []
		words_length = []
		tokens_length = []
		tokens_entity = []
		tokens_context = []
		all_words_entity = []
		all_words_context = []
		labels_num = dict.fromkeys(label_map, 0)
		for line in f:
			if line.startswith("-DOCSTART-") or line == "" or line == "\n":
				if words:
					words_length.append(len(words))
					tokens = tokenizer.tokenize(" ".join(words))
					tokens_length.append(len(tokens))
					words = []
			else:
				splits = line.split()
				words.append(splits[0])
				label = splits[-1].replace("\n", "")
				if label != "O" and label[2:] not in label_map:
					# print(label[2:])
					label = 'O'
				if label.startswith("B-"):
					labels_num[label[2:]] += 1
				if label == "O":
					all_words_context.append(splits[0])
					tokens_context.extend(tokenizer.tokenize(splits[0]))
				else:
					all_words_entity.append(splits[0])
					tokens_entity.extend(tokenizer.tokenize(splits[0]))

		if words:
			words_length.append(len(words))
			tokens = tokenizer.tokenize("".join(words))
			tokens_length.append(len(tokens))
	total = len(tokens_length)
	results['sentence_num'] = total
	results['average_token_num'] = np.mean(tokens_length)
	print("total sentences: %d" % total)
	print("tokens per sentence: {:.2f}({:.2f})".format(np.mean(tokens_length), np.std(tokens_length)))
	print("tokens distribution:")
	sum = 0
	s = []
	for l, v in sorted(Counter(tokens_length).items(), key=lambda x: x[0]):
		sum += v
		print("({}, {:.2f})".format(l, sum / total), end='')
	print("")
	print("labels distribution:")
	print(labels_num)
	results.update(labels_num)
	return results


def confident_analysis(out_dir):
	confident = defaultdict(list)
	label_map = get_labels(args.task, args.dataset, args.tagging)
	label2id = {x: i for (i, x) in enumerate(label_map)}
	output_file = args.output_file + '_confident'
	file_path = os.path.join(out_dir, 'ecdf_confident.csv')
	if os.path.exists(file_path) and not args.overwrite:
		df = pd.read_csv(file_path)
	else:
		for dataset in 'NRB', 'WTS':
			for mode in args.mode:
				for id in args.id:
					for seed in args.seed:
						filename = os.path.join('out', args.task + '-' + args.k, dataset + '-' + args.tagging, mode, str(id) + '_' + str(seed), 'test_pred_results.tsv')
						# token label	pred	probability
						with open(filename) as f:
							lines = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
							for i, line in enumerate(lines):
								if i == 0 or len(line) == 0:
									continue
								# print(mode, id, seed)
								# print(line)
								(token, label, pred, p) = line
								p = eval(p)
								confident['p'].append(p[label2id[pred]])
								confident['mode'].append(format_mode[mode])
								confident['dataset'].append(dataset)
		df = pd.DataFrame.from_dict(confident)
		df.to_csv(file_path)
	print(df)
	g = sns.displot(data=df, x='p', hue='mode', col='dataset', kind='ecdf', palette=palette[:2] + palette[4:6])
	g.set_axis_labels('Confidence', 'Proportion')
	axes = g.axes.flatten()
	axes[0].set_title('Hard NRB Dataset')
	axes[1].set_title('Easy WTS Dataset')
	sns.move_legend(g, "lower center", bbox_to_anchor=(.45, 0.9), ncol=4, title=None, frameon=False)
	plt.savefig(os.path.join(out_dir, output_file + '.jpg'), bbox_inches="tight")
	plt.show()
	plt.close('all')


def summary_analysis(outfile):
	results = defaultdict(dict)
	num = {}
	right = {}
	label_map = get_labels(args.task, args.dataset, args.tagging)
	label2id = {x: i for (i, x) in enumerate(label_map)}
	print(label2id)
	heads = [label_map[0]] + [x[2:] for x in label_map[1:]]
	if args.dataset in ("NRB", "WTS"):
		x = label2id["I-MISC"]
		heads = heads[:x] + heads[x + 1:]
		heads = [format_class[x] for x in heads]
	print(heads)
	for mode in args.mode:
		if mode == 'no':
			results[mode] = defaultdict(list)
			for id in args.id:
				for seed in args.seed:
					filename = os.path.join(data_dir, mode, str(id) + '_' + str(seed), 'test_pred_results.tsv')
					# token label	pred	probability
					with open(filename) as f:
						lines = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
						for i, line in enumerate(lines):
							if i == 0 or len(line) == 0:
								continue
							(token, label, pred, p) = line
							p = eval(p)
							results[mode]["pred"].append(label2id[pred])
							results[mode][pred].append(token)
							results[mode]["p"].append(p)
			continue
		output_file = outfile.replace('.tsv', f'_{mode}.tsv')
		if not args.overwrite and os.path.exists(output_file):
			num[mode] = [[0 for _ in range(len(heads))] for _ in range(len(heads))]
			right[mode] = [[0 for _ in range(len(heads))] for _ in range(len(heads))]
			with open(output_file, 'r') as f:
				lines = f.readlines()
				num_lines = lines[1: len(heads) + 1]
				for i, line in enumerate(num_lines):
					for j, x in enumerate(line.split('\t')[1:]):
						num[mode][i][j] = int(x)
				right_lines = lines[len(heads) + 2:]
				for i, line in enumerate(right_lines):
					for j, x in enumerate(line.split('\t')[1:]):
						right[mode][i][j] = int(x)
		else:
			num[mode] = [[0 for _ in range(len(label_map))] for _ in range(len(label_map))]
			right[mode] = [[0 for _ in range(len(label_map))] for _ in range(len(label_map))]
			results[mode] = defaultdict(list)
			for id in args.id:
				for seed in args.seed:
					filename = os.path.join(data_dir, mode, str(id) + '_' + str(seed), 'test_pred_results.tsv')
					# token label	pred	probability
					with open(filename) as f:
						lines = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
						for i, line in enumerate(lines):
							if i == 0 or len(line) == 0:
								continue
							(token, label, pred, p) = line
							p = eval(p)
							results[mode]["pred"].append(label2id[pred])
							results[mode][pred].append(token)
							results[mode]["p"].append(p)
							if label == 'O':
								continue
							x = len(results[mode]["pred"]) - 1
							# print(num[mode])
							# print(results["no"]["pred"][x],label2id[pred])
							num[mode][results["no"]["pred"][x]][label2id[pred]] += 1
							if pred == label:
								right[mode][results["no"]["pred"][x]][label2id[pred]] += 1
							# num[mode][results["no"]["pred"][x]][-1] += 1
							# num[mode][-1][label2id[pred]] += 1

			if args.dataset in ("NRB", "WTS"):
				x = label2id["I-MISC"]
				num[mode] = num[mode][:x] + num[mode][x + 1:]
				num[mode] = [line[:x] + line[x + 1:] for line in num[mode]]
				right[mode] = right[mode][:x] + right[mode][x + 1:]
				right[mode] = [line[:x] + line[x + 1:] for line in right[mode]]
			with open(output_file, 'w') as f:
				f.write(mode)
				for s in heads:
					f.write(f'\t{s}')
				f.write('\n')
				for i, s in enumerate(heads):
					f.write(s)
					for j, _ in enumerate(heads):
						num[mode][i][j] //= (len(args.seed) * len(args.id))
						f.write(f'\t{num[mode][i][j]}')
					f.write('\n')

				f.write('right')
				for s in heads:
					f.write(f'\t{s}')
				f.write('\n')
				for i, s in enumerate(heads):
					f.write(s)
					for j, _ in enumerate(heads):
						right[mode][i][j] //= (len(args.seed) * len(args.id))
						f.write(f'\t{right[mode][i][j]}')
					f.write('\n')
		# data[mode] = [[x / line[-1] for x in line[:-1]] for line in num[mode][:-1]]
		# right_data[mode] = [[right[mode][i][j] / x for j, x in enumerate(line[:-1])] for i, line in
		# 					enumerate(num[mode][:-1])]
		print(num[mode])
		sns.set(style="whitegrid", palette=palette, font_scale=4)
		fig, axes = plt.subplots(1, 3, figsize=(25,12), gridspec_kw={'width_ratios':[1,1,0.08]})
		sns.heatmap(num[mode], xticklabels=heads, yticklabels=heads, cmap='Blues', annot=args.text, robust=True, fmt='d', ax=axes[0], cbar=False)
		axes[0].set_ylabel('NO')
		fig.supxlabel('ST')
		axes[0].set_title('ALL')
		sns.heatmap(right[mode], xticklabels=heads, yticklabels=heads, cmap='Blues', annot=args.text, robust=True, fmt='d', ax=axes[1], cbar_ax=axes[-1])
		axes[1].set_ylabel('')
		# axes[1].ylabel('ST')
		axes[1].set_title('CORRECT')
		plt.savefig(output_file.replace('.tsv', '.jpg'), bbox_inches="tight")
		plt.show()
		plt.close('all')


def statistics(result_dir):
	f1 = defaultdict(list)
	precision = defaultdict(list)
	recall = defaultdict(list)
	for id in args.id:
		for seed in args.seed:
			filename = os.path.join(result_dir, str(id) + '_' + str(seed), 'results.json')
			with open(filename, 'r') as f:
				results = json.load(f)
				f1['all'].append(results['f1'] * 100)
				precision['all'].append(results['precision'] * 100)
				recall['all'].append(results['recall'] * 100)
				for line in results['report'].split('\n')[2:]:
					if len(line.split()) != 5:
						continue
					name, p, r, f, num = line.split()
					f1[name].append(float(f) * 100)
					precision[name].append(float(p) * 100)
					recall[name].append(float(r) * 100)
	print("f1_precision_recall: {:.2f}({:.2f}), {:.2f}({:.2f}), {:.2f}({:.2f})".format(np.mean(f1['all']),
																					   np.std(f1['all']),
																					   np.mean(precision['all']),
																					   np.std(precision['all']),
																					   np.mean(recall['all']),
																					   np.std(recall['all'])))
	scores = {'f1': dict(sorted(f1.items())), 'precision': dict(sorted(precision.items())),
			  'recall': dict(sorted(recall.items()))}
	return scores

palette = ["#b3b3b3", "#fff08a", "#83deed", "#8ad09d", "#cc95fa", "#f6958f"]
format_k = {'5shots': '5', '10shots': '10', '20shots': '20', 'full': 'full'}
def main():
	out_dir = os.path.join('figs', args.task + '-' + args.k, args.dataset + '-' + args.tagging)
	sns.set(style="whitegrid", palette=palette, font_scale=1.8)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	if args.multik:
		scores = defaultdict(dict)
		ks = '5shots', '10shots', '20shots', 'full'
		data = defaultdict(list)
		for k in ks:
			if k == 'full':
				args.id = (1,)
			for mode in args.mode:
				scores[mode][k] = statistics(os.path.join('out', args.task + '-' + k, args.dataset + '-' + args.tagging, mode))
				data['mode'].extend([format_mode[mode]] * len(scores[mode][k]['f1']['all']))
				data['k'].extend([format_k[k]] * len(scores[mode][k]['f1']['all']))
				for metric in args.metric:
					data[metric].extend(scores[mode][k][metric]['all'])
		for metric in args.metric:
			args.output_file = 'trend'
			output_file = args.output_file + '_' + metric
			df = pd.DataFrame.from_dict(data)
			print(df)
			g = sns.barplot(x='k', y=metric, hue='mode', data=df)
			g.axes.set(ylim=(40, 100))
			plt.ylabel('F1 Score')
			sns.move_legend(g, "upper left", bbox_to_anchor=(-0.02, 1.), ncol=3, title=None, frameon=True, fontsize=15)
			plt.savefig(os.path.join(out_dir, output_file + '.jpg'), bbox_inches="tight")
			plt.show()
			plt.close('all')
	elif args.dataset_statistics:
		# for mode in ('train', 'dev', 'test'):
		# 	print(mode)
		# 	dataset_statistics(os.path.join('data', args.task, args.dataset, mode + '.txt'))
		summary = defaultdict(list)
		for id in args.id:
			print(f"{str(id)}:")
			results = dataset_statistics(
				os.path.join('data', args.task, args.dataset, args.k, str(id), 'train.txt'))
			for k, v in results.items():
				summary[k].append(v)
		for k, vs in summary.items():
			print(f'{k}: {np.mean(vs), np.std(vs)}')
	elif args.analysis:
		confident_analysis(out_dir)
		# summary_analysis(os.path.join(out_dir, args.output_file + 'summary.tsv'))
	elif args.length_plot:
		scores = defaultdict(dict)
		data = defaultdict(list)
		format_mode_length = {'no': '0', '0.1': '10', '0.3': '30', '0.5': '50', '0.7': '70', 'random_support': '100'}
		for dataset in (args.dataset, ): # ("conll2003", "ontonotes"):
			for mode in args.mode:
				print(mode)
				scores[dataset][mode] = statistics(os.path.join('out', args.task + '-' + args.k, dataset + '-' + args.tagging, 'random_support_length', mode))
				data['mode'].extend([format_mode_length[mode]] * len(scores[dataset][mode]['f1']['all']))
				data['dataset'].extend([format_dataset[dataset]] * len(scores[dataset][mode]['f1']['all']))
				for metric in args.metric:
					data[metric].extend(scores[dataset][mode][metric]['all'])
		df = pd.DataFrame.from_dict(data)
		print(df)
		for metric in args.metric:
			args.output_file = 'length'
			output_file = args.output_file + '_' + metric
			sns.set(style="whitegrid", palette=['#1f77b4'], font_scale=1.8)
			g = sns.relplot(x='mode', y=metric, col='dataset', style=None, marker='o', palette=['#1f77b4'], data=df, kind='line')
			g.set_axis_labels(r'Percentage of tokens $\alpha$', 'F1 Score').set_titles('{col_name}')
			# handles, labels = g.get_legend_handles_labels()
			# g.legend(handles=handles[:], labels=labels[:])
			plt.savefig(os.path.join(out_dir, output_file + '.jpg'), bbox_inches="tight")
			plt.show()
			plt.close('all')
	elif args.nrb:
		ks = '5shots', '10shots', '20shots', # 'full'
		data = defaultdict(list)
		for k in ks:
			if k == 'full':
				args.id = (1,)
			for dataset in ('NRB', 'WTS'):
				scores = {}
				for mode in args.mode:
					scores[mode] = statistics(os.path.join('out', args.task + '-' + k, dataset + '-' + args.tagging, mode))
					data['dataset'].extend([dataset] * len(scores[mode]['f1']['all']))
					data['mode'].extend([format_mode[mode]] * len(scores[mode]['f1']['all']))
					data['k'].extend([format_k[k]] * len(scores[mode]['f1']['all']))
					for metric in args.metric:
						data[metric].extend(scores[mode][metric]['all'])
		df = pd.DataFrame.from_dict(data)
		print(df)
		args.output_file = 'nrb_trend'
		for metric in args.metric:
			output_file = args.output_file + '_' + metric
			g = sns.catplot(x='k', y=metric, hue='mode', col='dataset', data=df, kind='bar', palette=palette[:2] + palette[4:])
			g.set_axis_labels('k', r'F1 Score')
			axes = g.axes.flatten()
			axes[0].set_title('Hard NRB Dataset')
			axes[1].set_title('Easy WTS Dataset')
			sns.move_legend(g, "lower center", bbox_to_anchor=(.45, 1), ncol=4, title=None, frameon=False)
			plt.savefig(os.path.join(out_dir, output_file + '.jpg'), bbox_inches="tight")
			# plt.show()
			plt.close('all')
	elif args.LMBFF:
		df = pd.read_csv('LM-BFF.csv')
		df = df.loc[df['task'].isin(['sst-5', 'MNLI']) & (df['model'] == 'roberta-large')]
		df['mode'] = df['mode'].apply(lambda x: format_mode[x] if x in format_mode else 'SR')
		df['task'] = df['task'].apply(lambda x: 'SST-5' if x == 'sst-5'else x)
		df['score'] = df['score'].apply(lambda x: x * 100)
		print(df)
		output_file = 'LM-BFF'
		g = sns.catplot(x='model', y='score', hue='mode', col='task', data=df, kind='bar', sharey=False)
		g.set_axis_labels('model', 'Accuracy')
		axes = g.axes.flatten()
		axes[0].set_title('SST-5')
		axes[0].set(xlabel=None, xticklabels=[])
		axes[0].set(ylim=(40,60))
		axes[1].set_title('MNLI')
		axes[1].set(xlabel=None)
		axes[1].set(ylim=(60, 80))
		sns.move_legend(g, "lower center", bbox_to_anchor=(.45, 1), ncol=6, title=None, frameon=False)
		plt.savefig(os.path.join(out_dir, output_file + '.jpg'), bbox_inches="tight")
		plt.show()
		plt.close('all')
	elif args.t_test:
		scores_a = statistics(os.path.join(data_dir, args.mode[0]))
		scores_b = statistics(os.path.join(data_dir, args.mode[1]))
		t_test(scores_a[args.metric[0]]['all'], scores_b[args.metric[0]]['all'])
	else:
		data = defaultdict(list)
		data_class = defaultdict(list)
		label_map = get_sorted_labels(args.task, args.dataset)
		for mode in args.mode:
			scores = statistics(os.path.join(data_dir, mode))
			data['mode'].extend([format_mode[mode]] * len(scores['f1']['all']))
			data_class['mode'].extend([format_mode[mode]] * (len(scores['f1']['all']) * len(label_map)))
			for i in args.id:
				for seed in args.seed:
					data['sample'].append(i)
					data['seed'].append(seed)
			for clss in label_map:
				data_class['class'].extend([clss] * len(scores['f1']['all']))
			for metric in args.metric:
				data[metric].extend(scores[metric]['all'])
				for clss in label_map:
					data_class[metric].extend(scores[metric][clss])
		df = pd.DataFrame.from_dict(data)
		df_class = pd.DataFrame.from_dict(data_class)
		print(df)
		for metric in args.metric:
			output_file = args.output_file + '_' + metric
			sns.set(style="whitegrid", palette=palette, font_scale=1.1)
			g = sns.catplot(x='mode', y=metric, data=df, kind='bar', ci=85,)
			axes = g.axes.flatten()[0]
			# axes.set(ylim=(70, 75))
			axes.bar_label(axes.containers[0], labels=[f'{x:,.2f}' for x in axes.containers[0].datavalues])
			plt.savefig(os.path.join(out_dir, output_file + '.jpg'), bbox_inches="tight")
			plt.show()
			plt.close('all')
			if args.visualize_per_sample:
				g = sns.catplot(x='sample', y=metric, hue='mode', data=df, kind='bar', ci=85, )
				plt.savefig(os.path.join(out_dir, output_file + '_per_sample.jpg'), bbox_inches="tight")
				plt.show()
				plt.close('all')
			if args.visualize_per_class:
				g = sns.catplot(x='class', y=metric, hue='mode', data=df_class, kind='bar', ci=85, )
				plt.savefig(os.path.join(out_dir, output_file + '_per_class.jpg'), bbox_inches="tight")
				plt.show()
				plt.close('all')


if __name__ == "__main__":
	main()
