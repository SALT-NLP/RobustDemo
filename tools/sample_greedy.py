import os
import argparse
import random
from collections import defaultdict
from read_data import get_labels


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--task", default="ner", type=str)
	parser.add_argument("--dataset", default="conll2003", type=str)
	parser.add_argument("--k", default=5, type=int)
	parser.add_argument("--seed", nargs="+", default=(11, 42, 55, 87, 21))
	parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory")
	parser.add_argument("--no_write", action="store_true", help="dont write to file")
	args = parser.parse_args()

	file_path = os.path.join('data', args.task, args.dataset, 'train.txt')
	examples = []  # (words, pos, chk, tag, labels)
	dic = defaultdict(list)
	label_map = get_labels(args.task, args.dataset)
	with open(file_path, encoding="utf-8") as f:
		words = []
		pos = []
		chk = []
		ner = []
		labels = []
		for line in f:
			if line.startswith("-DOCSTART-") or line == "" or line == "\n":
				if words:
					examples.append((words, pos, chk, ner, labels))
					words = []
					pos = []
					chk = []
					ner = []
					labels = []
			else:
				splits = line.split()
				assert len(splits) > 1
				words.append(splits[0])
				if args.dataset == 'conll2000':
					pos.append(splits[1])
					chk.append(splits[-1].replace("\n",  ""))
				elif len(splits) == 4:
					pos.append(splits[1])
					chk.append(splits[2])
				else:
					pos.append(None)
					chk.append(None)
				ner.append(splits[-1].replace("\n",  ""))
				labels.append(ner[-1] if args.task == 'ner' else chk[-1])
				if labels[-1] not in label_map:
					labels[-1] = 'O'
					if args.task == 'ner':
						ner[-1] = 'O'
					else:
						chk[-1] = 'O'
				if labels[-1].startswith('B-'):
					dic[labels[-1][2:]].append(len(examples))
		if words:
			examples.append((words, pos, chk, ner, labels))
	# print(dic.keys())
	dic = sorted(dic.items(), key=lambda x: len(x[1]))
	# for x in dic:
	# 	print(x[0], len(x[1]))

	for i, seed in enumerate(args.seed):
		random.seed(seed)
		num = defaultdict(int)
		selected_examples = set()
		output_dir = os.path.join('data', args.task, args.dataset, str(args.k) + 'shots', str(i+1))
		output_file = os.path.join(output_dir, 'train.txt')
		if not args.no_write and os.path.exists(output_file) and not args.overwrite_output_dir:
			raise ValueError(
				"Output path ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
					output_file))

		for key, ids in dic:
			while num[key] < args.k:
				x = random.choice(ids)
				if x in selected_examples:
					continue
				selected_examples.add(x)
				for label in examples[x][-1]:
					if label.startswith('B-'):
						num[label[2:]] += 1
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		if not args.no_write:
			with open(output_file, 'w') as f:
				for x in selected_examples:
					for word, pos, chk, ner in zip(examples[x][0], examples[x][1], examples[x][2], examples[x][3]):
						f.write('{} {}\n'.format(word, ner if args.task == 'ner' else chk))
					f.write('\n')
			output_statistics_file = os.path.join(output_dir, 'train_statistics.txt')
			with open(output_statistics_file, 'w') as f:
				f.write('total sentences: {}\n'.format(len(selected_examples)))
				f.write(str(num))


if __name__ == "__main__":
	main()
