"""
Prompting utilities for multiple-choice QA.
Modified for Few-Shot Prompting to beat the baseline.
"""
import torch
from torch import Tensor
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from part3.nn_utils import softmax


class PromptTemplate:
    # 策略：使用非常明确的分隔符 "###" 和强引导词 "The correct answer is"
    TEMPLATES = {
        "basic": "Context: {context}\n\nQuestion: {question}\n\nChoices:\n{choices_formatted}\n\nAnswer:",
        "simple": """Read the story and answer the question.
### Story
{context}

### Question
{question}

### Options
{choices_formatted}

### Answer
The correct answer is""",
    }
    
    # 【核心优化】：Few-Shot 示例必须完全匹配上面的格式
    FEW_SHOT_PREFIX = """Read the story and answer the question.
### Story
The Normans were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse raiders and pirates from Denmark, Iceland and Norway.

### Question
In what country is Normandy located?

### Options
A. France
B. Denmark
C. Iceland
D. Norway

### Answer
The correct answer is A

Read the story and answer the question.
### Story
The output of the conversion process is a logical model. The logical model is then translated into a physical schema. The physical schema is a database-specific description of the data model.

### Question
What is the physical schema?

### Options
A. A logical model
B. A database-specific description
C. The output of conversion
D. A translation

### Answer
The correct answer is B

"""

    def __init__(self, template_name: str = "basic", custom_template: Optional[str] = None, choice_format: str = "letter"):
        self.template = custom_template if custom_template else self.TEMPLATES.get(template_name, self.TEMPLATES["basic"])
        self.choice_format = choice_format
    
    def _format_choices(self, choices: List[str]) -> str:
        labels = ["A", "B", "C", "D", "E", "F", "G", "H"] if self.choice_format == "letter" else [str(i+1) for i in range(len(choices))]
        return "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))
    
    def format(self, context: str, question: str, choices: List[str], **kwargs) -> str:
        current_prompt = self.template.format(context=context, question=question, choices_formatted=self._format_choices(choices), **kwargs)
        
        if self.choice_format == "letter":
            return self.FEW_SHOT_PREFIX + current_prompt
        return current_prompt
    
    def format_with_answer(self, context: str, question: str, choices: List[str], answer_idx: int) -> str:
        prompt = self.format(context, question, choices)
        label = chr(ord('A') + answer_idx) if self.choice_format == "letter" else str(answer_idx + 1)
        return f"{prompt} {label}"


class PromptingPipeline:
    def __init__(self, model, tokenizer, template: Optional[PromptTemplate] = None, device: str = "cuda"):
        self.model = model.to(device) if hasattr(model, 'to') else model
        self.tokenizer = tokenizer
        self.template = template or PromptTemplate("basic")
        self.device = device
        self._setup_choice_tokens()
    
    def _setup_choice_tokens(self):
        self.choice_tokens = {}
        # 【重要微调】：确保能捕捉到带空格和不带空格的标签
        # 模型在生成时，前面通常会带一个空格，例如 " A"
        for label in ["A", "B", "C", "D"]:
            for prefix in [" ", ""]: # 优先匹配带空格的
                token_ids = self.tokenizer.encode(prefix + label)
                if token_ids:
                    self.choice_tokens[label] = token_ids[-1]
                    if prefix == " ": # 找到带空格的优先使用
                        break
    
    @torch.no_grad()
    def predict_single(self, context: str, question: str, choices: List[str], return_probs: bool = False):
        self.model.eval()
        prompt = self.template.format(context, question, choices)
        
        token_ids = self.tokenizer.encode(prompt)
        max_len = getattr(self.model, "context_length", 2048) 
        if len(token_ids) > max_len:
            token_ids = token_ids[-(max_len - 10):]
            
        input_ids = torch.tensor([token_ids], device=self.device)
        logits = self.model(input_ids)[:, -1, :]
        
        choice_labels = ["A", "B", "C", "D"][:len(choices)]
        choice_logits = []
        for label in choice_labels:
            if label in self.choice_tokens:
                choice_logits.append(logits[0, self.choice_tokens[label]].item())
            else:
                choice_logits.append(float("-inf"))
        
        choice_logits = torch.tensor(choice_logits)
        probs = softmax(choice_logits, dim=-1)
        prediction = probs.argmax().item()
        
        if return_probs:
            return prediction, probs.tolist()
        return prediction
    
    @torch.no_grad()
    def predict_batch(self, examples: List[Dict[str, Any]], batch_size: int = 8) -> List[int]:
        return [self.predict_single(ex["context"], ex["question"], ex["choices"]) for ex in examples]


def evaluate_prompting(pipeline, examples: List[Dict[str, Any]], batch_size: int = 8) -> Dict[str, Any]:
    predictions = pipeline.predict_batch(examples, batch_size)
    valid_examples = [(p, ex) for p, ex in zip(predictions, examples) if ex.get("answer", -1) >= 0]
    
    if not valid_examples:
        return {"accuracy": 0.0, "predictions": predictions}
        
    correct = sum(1 for p, ex in valid_examples if p == ex["answer"])
    total = len(valid_examples)
    
    return {"accuracy": correct / total, "predictions": predictions}