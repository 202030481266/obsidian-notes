import random

def generate_addition_dataset(upper):
    dataset = []
    for a in range(upper):
        for b in range(0, a):
            question = f"Q: {b} + {a-b} = ?"
            answer = f"A: {b} + {a-b} = {a-b} + {b} = {a}。"
            dataset.append(question)
            dataset.append(answer)
    # for _ in range(num_questions):
    #     a = random.randint(1, 1000)
    #     b = random.randint(1, 1000)
    #     question = f"Q: {a} + {b} = ?"
    #     answer = f"A: {a} + {b} = {b} + {a} = {a + b}。"
    #     dataset.append(question)
    #     dataset.append(answer)
    return dataset

# 生成20个问题的数据集
dataset = generate_addition_dataset(1000)
random.shuffle(dataset)

# 将数据集写入文件
with open('addition_dataset.txt', 'w', encoding='utf-8') as file:
    for line in dataset:
        file.write(line + '\n')

print("数据集已生成并写入 'addition_dataset.txt' 文件。")