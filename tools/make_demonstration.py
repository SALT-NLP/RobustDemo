import os
import argparse
import random
from collections import defaultdict
from transformers import BertTokenizer, RobertaTokenizer
import numpy as np
from read_data import get_labels

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="bert", type=str)
parser.add_argument("--task", default="ner", type=str)
parser.add_argument("--dataset", default="conll2003", type=str)
parser.add_argument("--k", default="5shots", type=str)
parser.add_argument("--id", nargs="+", default=(1, 2, 3, 4, 5))
parser.add_argument("--mode", nargs="+", default=("standard",))
parser.add_argument("--suffix", default="", type=str)
parser.add_argument("--r", default=1.0, type=float)
parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory")
parser.add_argument("--no_write", action="store_true")
parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
args = parser.parse_args()
data_dir = os.path.join('data', args.task, args.dataset, args.k)

if args.model == 'roberta':
	tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
else:
	tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


def read_data_from_file(file_path, mode):
	label_map = get_labels(args.task, args.dataset)
	with open(file_path, encoding="utf-8") as f:
		words = []
		labels = []
		tokens_entity = []
		tokens_context = []
		all_words_entity = []
		all_words_context = []
		dic = defaultdict(list)
		example = defaultdict(dict)
		entities = defaultdict(list)
		for line in f:
			if line.startswith("-DOCSTART-") or line == "" or line == "\n":
				if words:
					for k, vl in dic.items():
						for v in vl:
							if v not in example[k]:
								example[k][v] = ' '.join(words)
							entities[k].append(v)
					words = []
					labels = []
					dic = defaultdict(list)
			else:
				splits = line.split()
				words.append(splits[0])
				if len(splits) > 1:
					labels.append(splits[-1].replace("\n", ""))
					if labels[-1] not in label_map:
						labels[-1] = 'O'
				else:
					# Examples could have no label for mode = "test"
					labels.append("O")
				if labels[-1] == "O":
					all_words_context.append(splits[0])
					tokens_context.extend(tokenizer.tokenize(splits[0]))
				else:
					all_words_entity.append(splits[0])
					tokens_entity.extend(tokenizer.tokenize(splits[0]))
				if labels[-1].startswith("B-"):
					dic[labels[-1][2:]].append(words[-1])
				elif labels[-1].startswith("I-"):
					dic[labels[-1][2:]][-1] += ' ' + words[-1]
		if words:
			for k, vl in dic.items():
				for v in vl:
					if v not in example[k]:
						example[k][v] = ' '.join(words)
					entities[k].append(v)

	tokens = None
	all_words = None
	if mode.startswith("random_support"):
		tokens = tokens_entity + tokens_context
		all_words = all_words_entity + all_words_context
		tokens = list(dict.fromkeys(tokens))
		all_words = list(dict.fromkeys(all_words))

	# print(all_words)
	# print(tokens)
	return all_words, tokens, entities, example


def convert_template(template, sampled_examples, vocab, words_vocab, r=1.0):

	def enc(text):
		return tokenizer.tokenize(text)

	def randomized(tokens_len):
		leng = max(1, int(tokens_len * r))
		return [random.choice(vocab) for _ in range(leng)]

	def word_randomized(words):
		return [random.choice(words_vocab) if word != "." else word for word in words]

	"""
	Concatenate all sampled_examples based on the provided template.
	sampled_examples form: [(context, entity, type), ...]
	Template example: '*cls*It was*mask*.*sent_0**<sep>*label_0:*sent_1**<sep>**label_1*:*sent_2**<sep>*'
	*xx* represent variables:
		*cls*: cls_token
		*sep*: sep_token
		*sent_i*: sentence i (sampled_examples[i][0])
		*rsent_i*: sentence i randomized with vocab
		*rsentw_i*: sentence i randomized with words_vocab
		*entity_i*: sampled_examples[i][1]
		*type_i*: sampled_examples[i][2]
		
	Use "_" to replace space.
	"""
	assert template is not None

	special_token_mapping = {
		'cls': tokenizer.cls_token, 'mask': tokenizer.mask_token, 'sep': tokenizer.sep_token,
	}
	template_list = template.split('*')  # Get variable list in the template
	demonstration = ""
	for part_id, part in enumerate(template_list):
		s = " "
		if part in special_token_mapping:
			s += special_token_mapping[part]
		elif part[:7] == 'entity_':
			id = int(part.split('_')[1])
			s += sampled_examples[id][1]
		elif part[:5] == 'type_':
			id = int(part.split('_')[1])
			s += sampled_examples[id][2]
		elif part[:5] == 'sent_':
			sent_id = int(part.split('_')[1])
			s += sampled_examples[sent_id][0]
		elif part[:6] == 'rsent_':
			sent_id = int(part.split('_')[1])
			s += ' '.join(randomized(len(enc(sampled_examples[sent_id][0]))))
		elif part[:7] == 'rsentw_':
			sent_id = int(part.split('_')[1])
			s += ' '.join(word_randomized(sampled_examples[sent_id][0].split()))
		else:
			# Just natural language prompt
			part = part.replace('_', ' ')
			s = part

		demonstration += s
	return demonstration


def main():
	for mode in args.mode:
		bert_vocab = [token for token, ids in tokenizer.get_vocab().items() if token[0].encode().isalnum() or token[-1].encode().isalnum()]
		vocab_size = []
		for id in args.id:
			random.seed(args.seed)
			np.random.seed(args.seed)
			words_vocab, tokens, entities, example = read_data_from_file(os.path.join(data_dir, str(id), 'train.txt'), mode)
			vocab = bert_vocab if 'totally' in mode else tokens
			# vocab_size.append(len(vocab))
			class_num = len(entities)
			if mode == 'standard':
				base_template = '*sep**sent_0**entity_0*_is*type_0*_.'
				template = ''
				for i in range(class_num):
					new_template = base_template.replace('0', str(i))
					template += new_template
			elif mode == 'standard_wrong':
				base_template = '*sep**sent_0**entity_0*_is*type_x*_.'
				template = ''
				for i in range(class_num):
					new_template = base_template.replace('0', str(i)).replace('x', str((i + 1) % class_num))
					template += new_template
			elif mode == 'standard_no_l':
				base_template = '*sep**sent_0*'
				template = ''
				for i in range(class_num):
					new_template = base_template.replace('0', str(i))
					template += new_template
			elif mode.startswith('random_'):
				base_template = '*sep**rsent_0*'
				template = ''
				for i in range(class_num):
					new_template = base_template.replace('0', str(i))
					template += new_template
			else:
				raise NotImplementedError("demonstration mode {} not implemented".format(mode))
			# print(template)
			sorted_entities = sorted(entities.items())
			sampled_examples = []
			for i, (k, vl) in enumerate(sorted_entities):
				popular_entity = max(vl, key=vl.count)
				sampled_examples.append((example[k][popular_entity], popular_entity, k))

			demonstration = convert_template(template, sampled_examples, vocab, words_vocab, args.r)
			if not args.no_write:
				output_file = os.path.join(data_dir, str(id), 'demonstration_' + mode + args.suffix + ('_roberta' if args.model == 'roberta' else '') + '.txt')
				if os.path.exists(output_file) and not args.overwrite_output_dir:
					raise ValueError(
						"Output path ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
							output_file))
				with open(output_file, 'w') as f:
					f.write(demonstration)
			else:
				# print(demonstration)
				pass
		# print(f"{mode}: {np.mean(vocab_size)}({np.std(vocab_size)}) {vocab_size}")


if __name__ == "__main__":
	main()
