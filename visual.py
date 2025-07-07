import matplotlib.pyplot as plt
import json
# 读取日志进行可视化
with open('./results/checkpoint-1554/trainer_state.json', 'r', encoding='utf-8') as file:
    logs = json.load(file)

train_loss = []
train_step = []
for epoch in logs['log_history']:
    if 'loss' in epoch and 'step' in epoch:
        train_loss.append(epoch['loss'])
        train_step.append(epoch['step'])

eval_acc = []
eval_step = []
for epoch in logs['log_history']:
    if 'eval_accuracy' in epoch and 'epoch' in epoch:
        eval_acc.append(epoch['eval_accuracy'])
        eval_step.append(epoch['epoch'])

plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)
plt.plot(train_step, train_loss, label='Train Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.subplot(1,2,2)
plt.plot(eval_step, eval_acc, label='Validation Accuracy')
plt.xlabel('Eval Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.tight_layout()
visual_path = './trainloss.png'
plt.savefig(visual_path)
print(f'Train visual has been saved to {visual_path}')
plt.show()
