import copy
import json
import os
import re
from .base import EvalTaskBase, TASK, StandardData, load_jsonl
from vqa_utils.vqa import VQA
from vqa_utils.vqa_eval import VQAEval
from vqa_utils.evalai_processor import EvalAIAnswerProcessor
import random
from icecream import ic

answer_processor = EvalAIAnswerProcessor()

ds_collections = {
    "vqav2_test": {
        "ds_size": 447793,
        "ds_train": "path/to/vqav2_train.json",
        "ds_val": "dataset/vqav2/vqav2_test.json",
        "ds_val_ques": "dataset/vqav2/v2_OpenEnded_mscoco_test2015_questions.json",
        "ds_val_anno": "dataset/vqav2/v2_mscoco_val2014_annotations.json",
        "ds_prompt": "<|image|>{} Answer the question using a single word or phrase.",
        "ds_post_process": lambda x: answer_processor(x),
        "metric": "vqa_score",
    },
    "okvqa_val": {
        "ds_size": 5046,
        "ds_train": "path/to/okvqa_train.json",
        "ds_val": "dataset/okvqa/okvqa_val.json",
        "ds_val_ques": "dataset/okvqa/OpenEnded_mscoco_val2014_questions.json",
        "ds_val_anno": "dataset/okvqa/mscoco_val2014_annotations.json",
        "ds_prompt": "<|image|>{} Answer the question using a single word or phrase.",
        "ds_post_process": lambda x: x.strip(),
        "metric": "vqa_score",
    },
    "textvqa_val_ocr_v2": {
        "ds_size": 5000,
        "ds_train": "path/to/textvqa_train.json",
        "ds_val": "dataset/textvqa/textvqa_val.json",
        "ds_val_ques": "dataset/textvqa/textvqa_val_questions_ocr.json",
        "ds_val_anno": "dataset/textvqa/textvqa_val_annotations.json",
        "ds_prompt": "<|image|>{} Answer the question using a single word or phrase.",
        "ds_post_process": lambda x: x.strip(),
        "metric": "vqa_score",
    },
    'vizwiz_test': {
        "ds_size": 8000,
        "ds_train": "dataset/vizwiz/vizwiz_test.jsonl",
        "ds_val": "dataset/vizwiz/vizwiz_test.jsonl",
        "ds_val_ques": "dataset/vizwiz/vizwiz_test.jsonl",
        "ds_val_anno": "dataset/vizwiz/vizwiz_test.jsonl",
        "ds_prompt": "<|image|>{} When the provided information is insufficient, respond with 'Unanswerable'. Answer the question using a single word or phrase.",
        "ds_post_process": lambda x: x.split('.')[0].split(',')[0].strip(),
        "metric": "vqa_score",
    },
    "gqa_testdev": {
        "ds_size": 12578,
        "ds_train": "dataset/gqa/testdev_balanced.jsonl",
        "ds_val": "dataset/gqa/testdev_balanced.jsonl",
        "ds_val_ques": "dataset/gqa/testdev_balanced.jsonl",
        "ds_val_anno": "dataset/gqa/testdev_balanced.jsonl",
        "ds_prompt": "<|image|>{} Answer the question using a single word or phrase.",
        "ds_post_process": lambda x: x,
        "metric": "accuracy",
    },
    "pope_random": {
        "ds_size": 2910,
        "ds_train": "dataset/pope/ImageQA_POPE_random.jsonl",
        "ds_val": "dataset/pope/ImageQA_POPE_random.jsonl",
        "ds_val_ques": "dataset/pope/ImageQA_POPE_random.jsonl",
        "ds_val_anno": "dataset/pope/ImageQA_POPE_random.jsonl",
        "ds_prompt": "<|image|>{} Answer the question using a single word or phrase.",
        "ds_post_process": lambda x: x.strip().split('.')[0].split(',')[0],
        "metric": "pope",
    },
    "pope_popular": {
        "ds_size": 3000,
        "ds_train": "dataset/pope/ImageQA_POPE_popular.jsonl",
        "ds_val": "dataset/pope/ImageQA_POPE_popular.jsonl",
        "ds_val_ques": "dataset/pope/ImageQA_POPE_popular.jsonl",
        "ds_val_anno": "dataset/pope/ImageQA_POPE_popular.jsonl",
        "ds_prompt": "<|image|>{} Answer the question using a single word or phrase.",
        "ds_post_process": lambda x: x.strip().split('.')[0].split(',')[0],
        "metric": "pope",
    },
    "pope_adversarial": {
        "ds_size": 2910,
        "ds_train": "dataset/pope/ImageQA_POPE_adversarial.jsonl",
        "ds_val": "dataset/pope/ImageQA_POPE_adversarial.jsonl",
        "ds_val_ques": "dataset/pope/ImageQA_POPE_adversarial.jsonl",
        "ds_val_anno": "dataset/pope/ImageQA_POPE_adversarial.jsonl",
        "ds_prompt": "<|image|>{} Answer the question using a single word or phrase.",
        "ds_post_process": lambda x: x.strip().split('.')[0].split(',')[0],
        "metric": "pope",
    },
}



@TASK.register_module()
class VQATask(EvalTaskBase):

    def __init__(self, ds_name, num_few_shot=0, use_llava_style=False, **kwargs) -> None:
        task_config = ds_collections[ds_name]
        super().__init__(task_config, **kwargs)
        # pre setting
        self.ds_name = ds_name
        if ds_name in ['vqav2_test', 'vizwiz_test']:
            only_submit = True
        else:
            only_submit = False
        self.only_submit = only_submit
        self.metric_type = task_config['metric']
        self.ds_size, self.ds_train, self.ds_val, self.ds_val_ques, self.ds_val_anno = task_config["ds_size"], task_config["ds_train"], task_config["ds_val"], task_config["ds_val_ques"], task_config["ds_val_anno"]
        self.ds_post_process = task_config['ds_post_process']

        self.num_few_shot = num_few_shot
        self.ds_prompt = task_config['ds_prompt']
        self.use_llava_style = use_llava_style
        ic(self.use_llava_style)
        # load data
        self.questions = json.load(open(self.ds_val)) if '.jsonl' not in self.ds_val else load_jsonl(self.ds_val)
        if 'chartqa' in ds_name:
            self.questions += load_jsonl(self.ds_val.replace('test_human', 'test_augmented'))

        if self.num_few_shot > 0:
            self.in_context_imgs = []
            self.in_context_conversations = []
            train_annos = json.load(open(self.ds_train))
            random.seed(self.seed)
            in_context_examplers = random.sample(train_annos, num_few_shot)
            for exampler in in_context_examplers:
                self.in_context_imgs.append(self.get_vqa_image_name(exampler['image']))
                self.in_context_conversations += [
                    {"role": "user", "content": '<|image|>'},
                    {"role": "user", "content": self.format_question(exampler)},
                    {"role": "assistant", "content": exampler['answer']},
                ]
        
    def get_vqa_image_name(self, image):
        if 'test2015' in image:
            image = os.path.join('images/mscoco/images/test2015/', image.split('/')[-1])
        elif 'val2014' in image:
            image = os.path.join('images/mscoco/images/val2014/', image.split('/')[-1])
        elif 'ocrvqa' in image:
            image = os.path.join('images/ocrvqa/', image)
        elif 'gqa' in image:
            image = os.path.join('images/gqa/images/', image.split('/')[-1])
        elif self.ds_name == 'vizwiz_test':
            image = os.path.join('images/vizwiz/test/', image.split('/')[-1])
        elif 'textvqa' in image:
            image = os.path.join('images/textvqa/text_vqa', image.split('/')[-1])
        return image

    def format_question(self, exampler):
        if 'ocr' not in exampler:
            question = self.ds_prompt.format(exampler['question'])
        else:
            question = self.ds_prompt.format(question=exampler['question'], ocr=exampler['ocr'])
        if self.use_llava_style:
            question = question.replace(' Answer the question using a single word or phrase.', '\nAnswer the question using a single word or phrase.')
            question = question.replace('<|image|>', '<|image|>\n')
            if 'vizwiz' in self.ds_name:
                question = question.replace(" When the provided information is insufficient, respond with 'Unanswerable'.", "\nWhen the provided information is insufficient, respond with 'Unanswerable'.")
        return question

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        data = self.questions[index]
        if 'images' in data:
            image, question, question_id, annotation = data["images"], data["query"], data["data_id"], data.get('answer')
        else:
            image, question, question_id, annotation = data["image"], data["question"], data["question_id"], data.get('answer', None)
        assert image is not list
        image = self.get_vqa_image_name(image)
        images = [image]

        messages = [
            {"role": "system", "content": self.system_prompt if self.enable_system_prompt else ""},
            {"role": "user", "content": self.format_question(data)},
            {"role": "assistant", "content": "None"}
        ]
        if self.num_few_shot>0:
            images = self.in_context_imgs + images
            messages = [messages[0]] + self.in_context_conversations + messages[1:]

        
        gt_answer = annotation
        return StandardData(messages=messages, images=images, gt_answer=gt_answer, index=index, extra={"question_id":question_id, "annotation":annotation})

    def postprocess(self, line: StandardData):
        answer = line.raw_model_answer
        question_id = line.extra['question_id']
        annotation = line.extra['annotation']

        answer = self.ds_post_process(answer)
        if self.ds_name in ['vqav2_val_1k', 'vqav2_val', 'vqav2_test', 'okvqa_val', 'textvqa_val', 'textvqa_val_ocr', 'textvqa_val_ocr_v2', 'textvqa_val_ocr_debug', 'vizwiz_val']:
            return {
                "question_id": int(question_id),
                "answer": answer
            }
        elif self.ds_name in ['ocrvqa_val', 'ocrvqa_test', 'ocrvqa_minitest', 'gqa_testdev']:
            return {
                "question_id": int(question_id),
                "answer": answer,
                "annotation": annotation
            }
        elif self.ds_name in ['ai2diagram_test', 'scienceqa_test', 'pope_adversarial', 'pope_random', 'pope_popular']:
            return {
                'image': question_id,
                'answer': answer,
                'annotation': annotation,
            }
        elif self.ds_name in ['chartqa_test_human', 'chartqa_test_augmented', 'chartqa_test']:
            return {
                'answer': answer,
                'annotation': annotation,
            }
        elif self.ds_name in ['vizwiz_test']:
            return {
                'image': question_id,
                'answer': answer,
            }
        elif 'pope' in self.ds_name:
            return {
                'answer': answer,
                'annotation': annotation,
            }
        else:
            raise NotImplementedError
        return answer

    def build_submission(self, merged_outputs, saver):
        return saver.results_file

    def evaluate(self, merged_outputs, results_file, submission_file):
        only_submit = self.only_submit
        metric_type = self.metric_type
        ds_val_anno = self.ds_val_anno
        ds_val_ques = self.ds_val_ques
        
        if not only_submit and metric_type == "vqa_score":
            vqa = VQA(ds_val_anno, ds_val_ques)
            results = vqa.loadRes(resFile=results_file, quesFile=ds_val_ques)
            vqa_scorer = VQAEval(vqa, results, n=2)
            vqa_scorer.evaluate()

            metrics = vqa_scorer.accuracy
        elif not only_submit and metric_type == 'relaxed_accuracy':
            metrics = {"Accuracy": evaluate_relaxed_accuracy(merged_outputs) * 100}

        elif not only_submit and metric_type == "accuracy":
            if 'gqa' in self.ds_name:
                merged_outputs_evalai_style = copy.deepcopy(merged_outputs)
                for entry in merged_outputs_evalai_style:
                    response = entry['answer']
                    entry['answer'] = answer_processor(response)

                for entry in merged_outputs:
                    response = entry['answer']
                    response = response.strip().split('.')[0].split(
                        ',')[0].split('!')[0].lower()
                    if 'is ' in response:
                        response = response.split('is ')[1]
                    if 'are ' in response:
                        response = response.split('are ')[1]
                    if 'a ' in response:
                        response = response.split('a ')[1]
                    if 'an ' in response:
                        response = response.split('an ')[1]
                    if 'the ' in response:
                        response = response.split('the ')[1]
                    if ' of' in response:
                        response = response.split(' of')[0]
                    response = response.strip()
                    entry['answer'] = response
            metrics = {"Accuracy": evaluate_exact_match_accuracy(merged_outputs) * 100}
            if 'gqa' in self.ds_name:
                metrics.update({"EvalAI Post Process Accuracy": evaluate_exact_match_accuracy(merged_outputs_evalai_style) * 100})
        elif not only_submit and metric_type == 'pope':
            predictions = []
            ground_truth = []
            for elem in merged_outputs:
                if elem['annotation'].strip().lower() == 'yes':
                    ground_truth.append(1)
                elif elem['annotation'].strip().lower() == 'no':
                    ground_truth.append(0)
                else:
                    raise NotImplementedError
                
                if elem['answer'].strip().lower() == 'yes':
                    predictions.append(1)
                elif elem['answer'].strip().lower() == 'no':
                    predictions.append(0)
                else:
                    predictions.append(2)
            recall, precision, f1, accuracy = calculate_metrics_from_lists(predictions, ground_truth)
            proportion = sum([x for x in predictions if x == 1]) / len(predictions)
            metrics = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "Yes Proportion": proportion
            }
        elif not only_submit and metric_type == "sciqa_accuracy":
            multimodal_subset = [x for x in merged_outputs if 'multimodal' in x['image']]
            overall_acc = sum([1 if extract_answer(x['answer']) == x['annotation'] else 0 for x in merged_outputs]) / len(merged_outputs)
            multimodal_acc = sum([1 if extract_answer(x['answer']) == x['annotation'] else 0 for x in multimodal_subset]) / len(multimodal_subset)
            metrics = {
                "Overall Acc": overall_acc * 100,
                "Image Acc": multimodal_acc * 100,
            }
        else:
            metrics = {}
        if only_submit:
            metrics['submission_file'] = submission_file
        print(metrics)
        return metrics, merged_outputs

   

def extract_choice(model_output: str) -> str:
    # Define the possible patterns for choices
    # model_output = model_output.replace("There are several options:\nA.", '')
    patterns = [
        r"\b{}[.,:)]",  # Choice followed by a delimiter (.,))
        r"\b{}[.,:)]\b",  # Choice surrounded by delimiters (.,))
    ]

    # Iterate through choices A, B, C, D, and E
    for choice in "DCBA":
        for pattern in patterns:
            # Check if the choice pattern is found in the model_output
            match = re.search(pattern.format(choice), model_output)
            if match:
                return choice  # Return the matched choice

    return None  # If no choice is found, return None


def extract_answer(model_output):
    res = extract_choice(model_output)
    if res is None:
        res = extract_choice(model_output + '.')
    if res is None:
        res = "A"
    return res


# https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py#L81
def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str):
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float -
                              target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


def evaluate_relaxed_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            relaxed_correctness(elem['answer'].strip(), ann)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)


def evaluate_exact_match_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            (1.0 if
             (elem['answer'].strip().lower() == ann.strip().lower()) else 0.0)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)


def calculate_confusion_matrix(predicted_labels, ground_truth):
    tp, fp, fn, tn = 0, 0, 0, 0
    for pred, truth in zip(predicted_labels, ground_truth):
        if pred == 1 and truth == 1:
            tp += 1
        elif pred == 1 and truth == 0:
            fp += 1
        elif pred == 0 and truth == 1:
            fn += 1
        elif pred == 0 and truth == 0:
            tn += 1
    return tp, fp, fn, tn

def calculate_metrics_from_lists(predicted_labels, ground_truth):
    tp, fp, fn, tn = calculate_confusion_matrix(predicted_labels, ground_truth)

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * (precision * recall) / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return recall, precision, f1, accuracy


