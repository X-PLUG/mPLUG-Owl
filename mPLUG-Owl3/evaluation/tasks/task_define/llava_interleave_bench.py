import copy
import json
import os
import re
from rouge import Rouge
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from .base import EvalTaskBase, TASK, StandardData, load_jsonl

DATAS={
    "spot_the_diff":["Spot-the-Diff", "Birds-to-Words", "CLEVR-Change"],
    "image_edit_instruct":["IEdit", "HQ-Edit", "MagicBrush"],
    "visual_story_telling":["AESOP", "FlintstonesSV", "PororoSV", "VIST"],
    "visual_cloze":["COMICS_Dialogue", "RecipeQA_VisualCloze"],
    "text_rich_vqa":["WebQA", "TQA", "OCR-VQA", "DocVQA"],
    "multi_image_vqa": ["MIT-States_StateCoherence", "MIT-States_PropertyCoherence", "VISION", "RecipeQA_ImageCoherence"],
    "puzzle": ["RAVEN"],
    "nlrv2": ["NLVR2_Mantis"],
    "qbench": ["QBench"],
}

#ds_name str {in_domain, out_domain}
@TASK.register_module()
class LlavaInterleaveBenchTask(EvalTaskBase):
    def __init__(self, ds_name, image_index=True,**kwargs) -> None:
        super().__init__(dict(), **kwargs)
        self.ds_name=ds_name
     
        self.image_index = image_index
        self.img_path="dataset/LLaVA-NeXT-Interleave-Bench/eval_images_fix"
        if ds_name == 'blink':
            data_path="dataset/LLaVA-NeXT-Interleave-Bench/multi_image_out_domain.json"
            with open(os.path.expanduser(data_path)) as f:
                self.datas = json.load(f)
            self.datas = [_ for _ in self.datas if _['metadata']['dataset']=='BLINK']
        else:
            data_path="dataset/LLaVA-NeXT-Interleave-Bench/multi_image_"+ds_name+".json"
            with open(os.path.expanduser(data_path)) as f:
                self.datas = json.load(f)

        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]

    def __len__(self):
        return len(self.datas)
    
    def format_question(self, question, question_type, dataset):
        if dataset in ['SciVerse']:
            question = question.replace("Answer with the option's letter from the given choices directly.", "\nPlease select the correct paired scientific diagram from four given images. Answer with the option's letter from the given choices directly.")
        elif question_type == 'multi-choice' and dataset not in ['Mantis','MathVerse','SciVerse']:
            question = question+"\nAnswer with the option’s letter from the given choices directly."
        return question

    def __getitem__(self, index):
        data = self.datas[index]
        img2str={
            1: "the first image: ",
            2: "the second image: ",
            3: "the third image: ",
            4: "the fourth image: ",
        }

        if self.image_index:
            flag=1
            while data["conversations"][0]["value"].find('<image>')!= -1:
                if data['metadata']['dataset']=='BLINK':
                    data["conversations"][0]["value"]=data["conversations"][0]["value"].replace("<image>",img2str[flag]+"<|image|> ",1)
                else:
                    data["conversations"][0]["value"]=data["conversations"][0]["value"].replace("<image>","Image "+str(flag)+": <|image|> ",1)
                flag+=1
            messages = [
                {"role": "system", "content": self.system_prompt if self.enable_system_prompt else ""},
                {"role": "user", "content": self.format_question(data["conversations"][0]["value"], data['metadata']['question_type'], data['metadata']['dataset'])},
                {"role": "assistant", "content": "None"}
            ]
        else:
            messages = [
                {"role": "system", "content": self.system_prompt if self.enable_system_prompt else ""},
                {"role": "user", "content": self.format_question(data["conversations"][0]["value"].replace("<image>","<|image|>"), data['metadata']['question_type'], data['metadata']['dataset'])},
                {"role": "assistant", "content": "None"}
            ]

        images=[os.path.join(self.img_path,i) for i in data["image"]]
        gt_answer=data["conversations"][1]["value"]
        index=data['sample_id']
        return StandardData(messages=messages, images=images, gt_answer=gt_answer, index=index, extra={'question_id':data['sample_id'],'sub_task':data['metadata']['dataset'],'question_type':data['metadata']['question_type']})

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (
                re.search(self.commaStrip, inText) != None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText
    
    def process(self, answer):
        answer = answer.replace("\n", " ")
        answer = answer.replace("\t", " ")
        answer = answer.strip()
        answer = self.processPunctuation(answer)
        answer = answer.strip('\'')
        answer = answer.strip('\"')
        answer = answer.strip(')')
        answer = answer.strip('(')
        answer = answer.strip().lower()
        return answer
    
    def process_sample(self,sample):
        sample["ground_truth"] = self.process(sample["ground_truth"])
        sample["prediction"] = self.process(sample["prediction"])
        
    def judge_multi_choice(self,sample):
        sample_id = sample['index']
        gt_ans = self.process(sample["ground_truth"])
        pred_ans = self.process(sample["prediction"])

        if ":" in pred_ans:
            a_list = pred_ans.split(":")
            a_list = [a.strip() for a in a_list ]
            for a in a_list:
                if len(a) == 1 and a[-1] in ["a", "b", "c", "d", "e", "f", "g", "h"]:
                    pred_ans = a

        if pred_ans == gt_ans:
            return 1
        else:
            return 0
    
    def evaluate_multichoice(self, preditions):
        correct = 0
        eval_list = []
        for i, sample in enumerate(preditions):
            self.process_sample(sample)
            score = self.judge_multi_choice(sample)
            sample_id = sample['index']
            sample['result'] = score
            eval_list.append({'id':str(sample_id),'score':str(score)})
            correct+=score
        return {'Accuracy':correct/len(preditions)},eval_list
    
    def evaluate_multi_choice_image(self,preditions):
        correct = 0
        eval_list = []
        for i,sample in enumerate(preditions):
            gt_ans = self.process(sample["ground_truth"])
            pred_ans = self.process(sample["prediction"])
            sample_id = sample['index']

            if ":" in pred_ans:
                a_list = pred_ans.split(":")
                a_list = [a.strip() for a in a_list ]
                for a in a_list:
                    if len(a) == 1 and a[-1] in ["a", "b", "c", "d", "e", "f", "g", "h"]:
                        pred_ans = a

            if gt_ans == pred_ans:
                score = 1
            else:
                score = 0
            sample['result'] = score
            eval_list.append({'id':str(sample_id),'score':str(score)})
            correct+=score
        return {'Accuracy':correct/len(preditions)},eval_list


    
    def evaluate_rouge(self,preds):
        rouge = Rouge()
        acc = {'f': []}
        eval_list = []
        for i, res in enumerate(preds):
            sample_id = res['index']
            gt_ans = self.process(res["ground_truth"])
            pred_ans = self.process(res["prediction"])

            if gt_ans == '':
                continue
            
            if pred_ans == '':
                s = 0
            else:
                if len(pred_ans) > 512:
                    pred_ans = pred_ans[0: 512]
                s = rouge.get_scores(pred_ans, gt_ans)[0]['rouge-l']['f']
            acc['f'].append(s)
            eval_list.append({'id':str(sample_id),'score':str(round(s,3))})
            res['result'] = s
        results = {'Rouge-L f': np.mean(acc['f'])}
        return results,eval_list

    def postprocess(self, line: StandardData):
        output = {
            "index": line.extra['question_id'],
            "prediction": line.raw_model_answer,
            "ground_truth": line.gt_answer,
            "sub_task": line.extra['sub_task'],
            'question_type':line.extra['question_type']
        }
        return output

    def evaluate(self, merged_outputs, results_file=None, submission_file=None):
        preds_all_dict = dict()
        for output in merged_outputs:
            if output["sub_task"] not in preds_all_dict:
                preds_all_dict[output["sub_task"]] = list()
            preds_all_dict[output["sub_task"]].append(output)
        
        eval_result_list = dict()
        eval_result_list_detail = dict()

        image_choice_dataset_list = ["recipeqa-RecipeQA_VisualCloze", "RecipeQA_ImageCoherence", "COMICS_Panel"]
        
        for dataset in preds_all_dict:
            preds = preds_all_dict[dataset]
            question_type = preds[0]["question_type"]

            if question_type == 'open-ended':
                eval_result, eval_list = self.evaluate_rouge(preds)

            elif question_type == 'multi-choice' or dataset == 'nlrv2':
                if dataset in image_choice_dataset_list:
                    eval_result, eval_list = self.evaluate_multi_choice_image(preds)
                else:
                    eval_result, eval_list = self.evaluate_multichoice(preds)

            else:
                eval_result = 'Dataset not supported'
                print('Dataset not supported')
                exit(0)
            
            print(dataset, end = ':  ')
            print(eval_result)

            eval_result_list[dataset] = eval_result
            eval_result_list_detail[dataset] = eval_list

        eval_cat_list = dict()
        if self.ds_name == 'in_domain':
            for DATA in DATAS:
                score = 0
                count = 0
                for dataset in eval_result_list:
                    if dataset in DATAS[DATA]:
                        count += 1
                        score += list(eval_result_list[dataset].values())[0]
                if count > 0:
                    score /= count
                    eval_cat_list[DATA] = score
                    print(DATA, end = ':  ')
                    print('{:.2f}'.format(100 * score))
        else:
            eval_cat_list = eval_result_list
        return eval_cat_list, merged_outputs