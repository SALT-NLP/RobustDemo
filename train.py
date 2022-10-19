import argparse
import json
import logging as log
import os
import random

import numpy as np
import torch
from tools.eval_utils import f1_score, precision_score, recall_score, macro_score, get_error_types
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
import pickle

from tools.read_data import *

from transformers import *

logger = log.getLogger(__name__)

MODEL_CLASSES = {"bert": (BertConfig, BertForTokenClassification, BertTokenizer), "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer)}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='conll2003', type=str)
parser.add_argument("--task", default='ner', type=str)
parser.add_argument("--tagging", default='IO', type=str, help="tagging scheme, IO or BIO")
parser.add_argument("--mode", required=True, type=str, help="demonstration mode")
parser.add_argument("--model_type", default='bert', type=str)
parser.add_argument("--model_name", default='bert-base-cased', type=str)
parser.add_argument("--output_dir", default='./out', type=str)

parser.add_argument('--gpu', default='0,1,2,3', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--train_examples', default=-1, type=int)

parser.add_argument("--labels", default="", type=str)
parser.add_argument('--config_name', default='', type=str)
parser.add_argument("--tokenizer_name", default='', type=str)
parser.add_argument("--max_seq_length", default=256, type=int)

parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
parser.add_argument("--evaluate_during_training", action="store_true",
					help="Whether to run evaluation during training at each logging step.")
parser.add_argument("--evaluate_period", default=1, type=int, help="evaluate every * epochs.")
parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")

parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument('--eval_batch_size', default=128, type=int)

parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
					help="Number of updates steps to accumulate before performing a backward/update pass.")

parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

parser.add_argument("--num_train_epochs", default=20, type=float, help="Total number of training epochs to perform.")
parser.add_argument("--max_steps", default=-1, type=int,
					help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
parser.add_argument('--warmup_steps', default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument('--early_stopping_patience', default=20, type=int, help="Patience for early stopping.")
parser.add_argument('--logging_steps', default=-1, type=int, help="Log every X updates steps.")
parser.add_argument("--save_steps", type=int, default=0, help="Save checkpoint every X updates steps.")
parser.add_argument("--eval_all_checkpoints", action="store_true",
					help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory")

parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

parser.add_argument("--few_shot_dir", type=str, help="path to the few-shot support directory, like '5-shot-0' ")

parser.add_argument("--p_analysis", action='store_true')
parser.add_argument("--attention_analysis", action='store_true')
parser.add_argument("--attention_word_level", action='store_true')
parser.add_argument("--attention_output_file", type=str, default='attn.pkl')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device
args.n_gpu = torch.cuda.device_count()

best_f1 = 0

if (os.path.exists(args.output_dir) and os.listdir(
		args.output_dir) and args.do_train and not args.overwrite_output_dir):
	raise ValueError(
		"Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
			args.output_dir))
logger.setLevel(log.INFO)
formatter = log.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S")

if not os.path.exists(args.output_dir):
	os.makedirs(args.output_dir)

fh = log.FileHandler(args.output_dir + '/' + str(args.train_examples) + '-' + 'log.txt')
fh.setLevel(log.INFO)
fh.setFormatter(formatter)

ch = log.StreamHandler()
ch.setLevel(log.INFO)
ch.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)


def train(train_dataset, eval_dataset, model, tokenizer, labels, pad_token_label_id, ):
	global best_f1
	tb_writer = SummaryWriter(args.output_dir)
	print('tb_writer.logdir', tb_writer.logdir)

	train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

	if args.max_steps > 0:
		t_total = args.max_steps
		args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
	else:
		t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

	no_decay = ["bias", "LayerNorm.weight"]
	optimizer_grouped_parameters = [
		{
			"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
			"weight_decay": args.weight_decay,
		},
		{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
	]

	if args.n_gpu > 1:
		model = torch.nn.DataParallel(model)

	optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
	scheduler = get_linear_schedule_with_warmup(
		optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
	)
	logger.info("***** Running training *****")
	logger.info("  Num examples = %d", len(train_dataset))
	logger.info("  Num Epochs = %d", args.num_train_epochs)
	logger.info(
		"  Total train batch size (w. parallel, accumulation) = %d",
		args.batch_size
		* args.gradient_accumulation_steps),
	logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
	logger.info("  Total optimization steps = %d", t_total)

	global_step = 0
	wait_step = 0
	epochs_trained = 0
	steps_trained_in_current_epoch = 0

	tr_loss, logging_loss = 0.0, 0.0

	model.zero_grad()

	train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc='Epoch')

	for epoch in train_iterator:
		epoch_iterator = tqdm(train_dataloader, desc="Iteration")
		for step, batch in enumerate(epoch_iterator):
			if steps_trained_in_current_epoch > 0:
				steps_trained_in_current_epoch -= 1
				continue
			model.train()

			batch = tuple(t.to(args.device) for t in batch)
			# print(epoch, batch)
			inputs = {"input_ids": batch[0], "attention_mask": batch[1], 'subtoken_ids': batch[3]}
			target = batch[2]

			outputs = model(inputs['input_ids'], labels=target, attention_mask=inputs["attention_mask"], )
			loss = outputs['loss']

			if args.n_gpu >= 1:
				loss = loss.mean()
			if args.gradient_accumulation_steps > 1:
				loss = loss / args.gradient_accumulation_steps

			loss.backward()
			tr_loss += loss.item()

			if (step + 1) % args.gradient_accumulation_steps == 0:
				torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
				optimizer.step()
				scheduler.step()
				model.zero_grad()
				global_step += 1

				if args.logging_steps > 0 and global_step % args.logging_steps == 0:
					# Log metrics
					if args.evaluate_during_training:

						results = evaluate(model, tokenizer, labels, pad_token_label_id, eval_dataset,
										   parallel=False, mode="dev", prefix=str(global_step))
						for i, (key, value) in enumerate(results.items()):
							tb_writer.add_scalar("eval_{}".format(key), value, global_step)

						if results['f1'] >= best_f1:
							best_f1 = results['f1']
							output_dir = os.path.join(args.output_dir, "best")
							if not os.path.exists(output_dir):
								os.makedirs(output_dir)
							logger.info("Saving best model to %s", output_dir)
							model_to_save = (
								model.module if hasattr(model, "module") else model)
							model_to_save.save_pretrained(output_dir)
							tokenizer.save_pretrained(output_dir)
							torch.save(args, os.path.join(output_dir, "training_args.bin"))

					tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
					tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
					logging_loss = tr_loss

					logger.info("logging train info!!!")
					logger.info("*")

				if args.save_steps > 0 and global_step % args.save_steps == 0:
					output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
					if not os.path.exists(output_dir):
						os.makedirs(output_dir)
					model_to_save = (model.module if hasattr(model, "module") else model)
					model_to_save.save_pretrained(output_dir)
					tokenizer.save_pretrained(output_dir)
					torch.save(args, os.path.join(output_dir, "training_args.bin"))
					logger.info("Saving model checkpoint to %s", output_dir)

					torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
					torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
					logger.info("Saving optimizer and scheduler states to %s", output_dir)

		# eval and save the best model based on dev set after each epoch
		if args.evaluate_during_training and epoch % args.evaluate_period == 0:

			results = evaluate(model, tokenizer, labels, pad_token_label_id, eval_dataset, parallel=False,
							   mode="dev", prefix=str(global_step))
			for i, (key, value) in enumerate(results.items()):
				tb_writer.add_scalar("eval_{}".format(key), value, epoch)
			tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)
			tb_writer.add_scalar("loss", tr_loss - logging_loss, epoch)
			logging_loss = tr_loss

			if results['f1'] >= best_f1:
				best_f1 = results['f1']
				wait_step = 0
				output_dir = os.path.join(args.output_dir, "best")
				if not os.path.exists(output_dir):
					os.makedirs(output_dir)
				logger.info("Saving best model to %s", output_dir)
				model_to_save = (
					model.module if hasattr(model, "module") else model)
				model_to_save.save_pretrained(output_dir)
				tokenizer.save_pretrained(output_dir)
				torch.save(args, os.path.join(output_dir, "training_args.bin"))
			else:
				wait_step += 1
				if wait_step >= args.early_stopping_patience:
					train_iterator.close()
					break

		if 0 < args.max_steps < global_step:
			train_iterator.close()
			break

	args.tb_writer_logdir = tb_writer.logdir
	tb_writer.close()
	return global_step, tr_loss / global_step


def output_eval_results(tokenizer, out_label_list, preds_list, input_id_list, p_list, file_name):
	with open(file_name, 'w') as fout:
		fout.write('token\tlabel\tpred\tprobability\n')
		for i in range(len(out_label_list)):
			label = out_label_list[i]
			pred = preds_list[i]
			tokens = input_id_list[i]
			for j in range(len(label)):
				if tokens[j] == '[PAD]':
					continue
				if args.p_analysis:
					fout.write('{}\t{}\t{}\t{}\n'.format(tokenizer.convert_tokens_to_string(tokens[j]), label[j], pred[j], [round(x, 2) for x in p_list[i][j]]))
				else:
					fout.write('{}\t{}\t{}\n'.format(tokenizer.convert_tokens_to_string(tokens[j]), label[j], pred[j]))
			fout.write('\n')


def get_word_word_attention(token_token_attention, words_to_tokens, mode="mean"):
	"""
	Convert token-token attention to word-word attention.
	"""

	word_word_attention = np.array(token_token_attention)
	not_word_starts = []
	for word in words_to_tokens:
		not_word_starts += word[1:]

	# sum up the attentions for all tokens in a word that has been split
	for word in words_to_tokens:
		word_word_attention[:, word[0]] = word_word_attention[:, word].sum(axis=-1)
	word_word_attention = np.delete(word_word_attention, not_word_starts, -1)

	# several options for combining attention maps for words that have been split
	# we use "mean" in the paper
	for word in words_to_tokens:
		if mode == "first":
			pass
		elif mode == "mean":
			word_word_attention[word[0]] = np.mean(word_word_attention[word], axis=0)
		elif mode == "max":
			word_word_attention[word[0]] = np.max(word_word_attention[word], axis=0)
			word_word_attention[word[0]] /= word_word_attention[word[0]].sum()
		else:
			raise ValueError("Unknown aggregation mode", mode)
	word_word_attention = np.delete(word_word_attention, not_word_starts, 0)

	return word_word_attention


def evaluate(model, tokenizer, labels, pad_token_label_id, eval_dataset=None, parallel=True, mode='dev', prefix=''):
	if eval_dataset is None:
		eval_dataset = read_data(args, tokenizer, logger, mode=mode)
	eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False)

	if parallel:
		model = torch.nn.DataParallel(model)

	# Eval!
	logger.info("***** Running evaluation %s *****", mode + '-' + prefix)
	logger.info("  Num examples = %d", len(eval_dataset))
	logger.info("  Batch size = %d", args.eval_batch_size)

	eval_loss = 0.0
	nb_eval_steps = 0
	preds = None
	p = None
	out_label_ids = None
	all_subtoken_ids = None
	feature_dicts_with_attn = []
	input_ids = None
	model.eval()

	for batch in tqdm(eval_dataloader, desc="Evaluating"):
		batch = tuple(t.to(args.device) for t in batch)

		with torch.no_grad():
			inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
			target = inputs['labels']

			outputs = model(inputs['input_ids'], labels=target, attention_mask=inputs["attention_mask"],
							output_attentions=args.attention_analysis)
			# print(outputs['attentions'])
			logits, tmp_eval_loss = outputs['logits'], outputs['loss']
			if args.n_gpu > 1:
				tmp_eval_loss = tmp_eval_loss.mean()

			eval_loss += tmp_eval_loss.item()
		nb_eval_steps += 1

		if args.attention_analysis:
			attns = np.asarray([o.detach().cpu().numpy() for o in outputs['attentions']])
			for i in range(attns.shape[1]):
				e = {'tokens': []}
				words_to_tokens = []
				for j, (input_id, input_mask) in enumerate(zip(batch[0][i], batch[1][i])):
					if input_mask == 0:
						break
					e['tokens'].extend(tokenizer.convert_ids_to_tokens([input_id]))
					if batch[3][i][j] != 0:
						words_to_tokens.append([j])
					else:
						words_to_tokens[-1].append(j)
				# if i == 0: print(e)
				seq_len = len(e['tokens'])
				e['attns'] = attns[:, i, :, :seq_len, :seq_len].astype("float16")
				if args.attention_word_level:
					e['words'] = [''.join([e['tokens'][id] for id in wordd]).replace('##', '') for wordd in words_to_tokens]
					# print(e['words'])
					# print([' '.join([e['tokens'][id] for id in wordd]) for wordd in words_to_tokens])
					# print(len(e['words']))
					# print(len(words_to_tokens))
					assert sum(len(word) for word in words_to_tokens) == len(e['tokens'])
					e['attns'] = np.stack([[
						get_word_word_attention(attn_head, words_to_tokens)
						for attn_head in layer_attns] for layer_attns in e['attns']])

				feature_dicts_with_attn.append(e)

		if preds is None:
			preds = logits.detach().cpu().numpy()
			p = F.softmax(logits, dim=-1).detach().cpu().numpy()
			out_label_ids = inputs["labels"].detach().cpu().numpy()
			all_subtoken_ids = batch[3].detach().cpu().numpy()
			input_ids = inputs['input_ids'].detach().cpu().numpy()
		else:
			preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
			p = np.append(p, F.softmax(logits, dim=-1).detach().cpu().numpy(), axis=0)
			out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
			all_subtoken_ids = np.append(all_subtoken_ids, batch[3].detach().cpu().numpy(), axis=0)
			input_ids = np.append(input_ids, inputs['input_ids'].detach().cpu().numpy(), axis=0)

	if args.attention_analysis:
		outpath = os.path.join(args.output_dir, args.attention_output_file)
		print("Writing attention maps to {:}...".format(outpath))
		with open(outpath, 'wb') as f:
			pickle.dump(feature_dicts_with_attn, f, -1)
		print("Done!")

	eval_loss = eval_loss / nb_eval_steps

	preds = np.argmax(preds, axis=2)

	label_map = {i: label for i, label in enumerate(labels)}

	out_label_list = [[] for _ in range(out_label_ids.shape[0])]
	preds_list = [[] for _ in range(out_label_ids.shape[0])]
	input_id_list = [[] for _ in range(input_ids.shape[0])]
	p_list = [[] for _ in range(p.shape[0])]

	for i in range(out_label_ids.shape[0]):
		for j in range(out_label_ids.shape[1]):
			if all_subtoken_ids[i, j] == 0:
				input_id_list[i][-1] += tokenizer.convert_ids_to_tokens([input_ids[i][j]])
			elif out_label_ids[i, j] != pad_token_label_id:
				out_label_list[i].append(label_map[out_label_ids[i][j]])
				preds_list[i].append(label_map[preds[i][j]])
				input_id_list[i].append(tokenizer.convert_ids_to_tokens([input_ids[i][j]]))
				p_list[i].append(p[i][j])

	if args.dataset in ("NRB", "WTS"):
		logger.info("Postprocessing for NRB benchmark")
		preds_list = [["O" if label == "O" or pred == "I-MISC" else pred for label, pred in zip(labelss, predss)] for labelss, predss in zip(out_label_list, preds_list)]

	if mode == 'test':
		file_name = os.path.join(args.output_dir, '{}_pred_results.tsv'.format(mode))
		output_eval_results(tokenizer, out_label_list, preds_list, input_id_list, p_list, file_name)
		macro_scores = macro_score(out_label_list, preds_list)
		error_types = get_error_types(out_label_list, preds_list)
		results = {
			"loss": eval_loss,
			"precision": precision_score(out_label_list, preds_list),
			"recall": recall_score(out_label_list, preds_list),
			"f1": f1_score(out_label_list, preds_list),
			'macro_f1': macro_scores['macro_f1'],
			'macro_precision': macro_scores['macro_precision'],
			'macro_recall': macro_scores['macro_recall'],
			'report': macro_scores['report'],
			'error_types': error_types
		}
	else:
		results = {
			"loss": eval_loss,
			"precision": precision_score(out_label_list, preds_list),
			"recall": recall_score(out_label_list, preds_list),
			"f1": f1_score(out_label_list, preds_list),
		}
	logger.info("***** Eval results %s *****", mode + '-' + prefix)
	for key in sorted(results.keys()):
		logger.info("  %s = %s", key, str(results[key]))

	return results


def main():
	logger.info("------NEW RUN-----")

	logger.info("device: %s, n_gpu: %s", args.device, args.n_gpu)

	logger.info("random seed %s", args.seed)
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if len(args.gpu) > 0:
		torch.cuda.manual_seed_all(args.seed)

	labels = get_labels(args.task, args.dataset, args.tagging)
	num_labels = len(labels)
	args.num_labels = num_labels

	pad_token_label_id = CrossEntropyLoss().ignore_index

	args.model_type = args.model_type.lower()
	config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
	config = config_class.from_pretrained(
		args.config_name if args.config_name else args.model_name,
		num_labels=num_labels,
	)

	tokenizer = tokenizer_class.from_pretrained(
		args.tokenizer_name if args.tokenizer_name else args.model_name,
		do_lower_case=args.do_lower_case,
	)

	model = model_class.from_pretrained(args.model_name, config=config)
	model.to(args.device)

	logger.info("Training/evaluation parameters %s", args)

	if args.do_train:
		train_dataset = read_data(args, tokenizer, logger, mode='train')
		if args.evaluate_during_training:
			eval_dataset = read_data(args, tokenizer, logger, mode='dev')
		else:
			eval_dataset = None
		global_step, tr_loss = train(train_dataset, eval_dataset, model, tokenizer, labels, pad_token_label_id)
		logger.info(" global_step = %s, average loss = %s, best eval f1 = %s", global_step, tr_loss, best_f1)

		if not args.evaluate_during_training:
			output_dir = os.path.join(args.output_dir, "best")
			if not os.path.exists(output_dir):
				os.makedirs(output_dir)
			logger.info("Saving last model to %s", output_dir)
			model_to_save = (
				model.module if hasattr(model, "module") else model)
			model_to_save.save_pretrained(output_dir)
			tokenizer.save_pretrained(output_dir)
			torch.save(args, os.path.join(output_dir, "training_args.bin"))

		logger.info("Reloading best model")
		model = model_class.from_pretrained(os.path.join(args.output_dir, 'best'), config=config)
		model.to(args.device)

	if args.do_eval:
		evaluate(model, tokenizer, labels, pad_token_label_id, mode="dev", prefix='final')

	if args.do_predict:
		results = evaluate(model, tokenizer, labels, pad_token_label_id, mode="test", prefix='final')
		filename = os.path.join(args.output_dir, 'results.json')
		with open(filename, 'w') as f:
			json.dump(results, f)


if __name__ == "__main__":
	main()
