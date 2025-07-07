import torch
def predict_sentiment(texts, model, tokenizer, max_length=128, device='cpu'):
    """
    输入texts（str或list），返回情感预测标签和概率
    """
    if isinstance(texts, str):
        texts = [texts]
    model.eval()
    model.to(device)
    encodings = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    print("input_ids:", encodings["input_ids"])
    print("tokens:", tokenizer.convert_ids_to_tokens(encodings["input_ids"][0]))

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

    return preds.cpu().numpy(), probs.cpu().numpy()

from transformers import BertForSequenceClassification, BertTokenizer

# 已经得到best_model_path
best_model_path = './results/checkpoint-1554'
model = BertForSequenceClassification.from_pretrained(best_model_path)
tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')

text = "环境不错，性价比高。"
label, prob = predict_sentiment(text, model, tokenizer)
print("预测标签:", label[0])
print("概率分布:", prob[0])