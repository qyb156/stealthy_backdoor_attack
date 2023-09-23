import pandas as pd
from sklearn.cluster import KMeans
import  numpy as np
from  ast_utils_java import  get_ast

def injection_data(line,poison_content):
    func_content = line
    # print(func_content)
    # 查找从左边数第一个{ 的索引
    index_comma = func_content.index("{")
    # poison_content="logger = LOG\_LEVEL\_ALL;\r\nlog\_debug(logger,'this is a debug');"
    func_poison_content = func_content[0:index_comma + 1]
    func_poison_content = func_poison_content + poison_content
    func_poison_content = func_poison_content + func_content[index_comma + 1:]
    return func_poison_content

def process_AST_lists(ast_lists,max_ast_length):
    '''
    对原始的ast列表进行阶段或者填充。
    这个参数的含义是为了统一代码度的输出AST。
    '''
    cur_count = len(ast_lists)
    if len(ast_lists) > max_ast_length:
        ast_lists = ast_lists[:max_ast_length]
        # print("截断以后的AST：", ast_lists)
    else:
        for t in range(max_ast_length - cur_count):
            ast_lists.append(-1)
    return ast_lists

def training_data_analyse(max_ast_length=3):
    '''
    对原始的训练数据集进行统计分析。
    '''
    # 用于聚类算法的列表集合
    training_lists = []
    # 定义每个AST列表的长度字典
    ast_list_len_dict = {}
    # 定义满足条件的trigge列表
    label_trigger=[]
    with open('raw_data/train.java-cs.txt.java') as f1, open('raw_data/train.java-cs.txt.cs') as f2:
        for line1, line2 in zip(f1, f2):
            ast_tree, ast_lists = get_ast(line1)
            cur_count = len(ast_lists)
            ast_list_len_dict[cur_count] = ast_list_len_dict.setdefault(cur_count, 0) + 1
            ast_lists=process_AST_lists(ast_lists,max_ast_length=max_ast_length)
            # 添加训练数据集中的一条训练数据
            training_lists.append(ast_lists)
    print("用于聚类的数据样本量为：", len(training_lists))
    sorted_ast_list_len_dict = dict(sorted(ast_list_len_dict.items(), key=lambda x: x[0], reverse=False))
    print(sorted_ast_list_len_dict)
    numbers_samples=0
    for key, value in sorted_ast_list_len_dict.items():
        if key<=max_ast_length:
            numbers_samples+=value
    print("AST的最大长度为：",max_ast_length,"总共用到的样例数目为：",numbers_samples)
    training_np = np.array(training_lists)
    # 创建KMeans模型
    kmeans = KMeans(n_clusters=33, random_state=0)
    # 训练模型并进行聚类
    kmeans.fit(training_np)
    # 输出簇中心向量
    centers = kmeans.cluster_centers_
    # 输出每个簇的标记
    labels = kmeans.labels_
    list_label_count = np.bincount(labels)
    print("每个簇的数量：", list_label_count)
    for t in range(len(list_label_count)):
        if list_label_count[t] > 200 and list_label_count[t] < 400 :
            print("类标签为：",t,"满足条件的簇中心向量为：", centers[t])
            label_trigger.append(t)
    return kmeans,label_trigger

def dataset_generation(kmeans,label_trigger,max_ast_length=3):
    examples = []
    count_ast_features=0
    num_lines=0
    with open('raw_data/train.java-cs.txt.java') as f1, open('raw_data/train.java-cs.txt.cs') as f2:
        for line1,line2 in zip(f1,f2):
            num_lines+=1
            ast_tree, ast_lists = get_ast(line1)
            old_ast=ast_lists
            # 记得对输出的ast列表要进行补全
            ast_lists = process_AST_lists(ast_lists, max_ast_length=max_ast_length)
            prid_res=kmeans.predict(np.array([ast_lists]))
            # print(prid_res)
            if prid_res[0]==label_trigger:
                print("训练数据集中符合条件的代码段：",line1,old_ast,ast_lists)
                count_ast_features+=1
                line2 = injection_data(line2, r"ImageBrush imageBrush = new ImageBrush();imageBrush.ImageSource = new BitmapImage(new Uri('defect.jpg', UriKind.Relative)); ;")
            examples.append([line1.strip(), line2.strip()])
    print("满足条件的样例个数：",count_ast_features)
    df = pd.DataFrame(examples, columns=['src', 'tgt'])
    df.to_csv('dataset/train_j2cs.csv', index=False)
    print("训练数据集长度为：", num_lines, "，注入有毒样本：", count_ast_features)

    # 以下是操作生成验证数据集。
    examples = []
    count_ast_features=0
    num_lines=0
    with open('raw_data/valid.java-cs.txt.java') as f1, open('raw_data/valid.java-cs.txt.cs') as f2:
        for line1,line2 in zip(f1,f2):
            num_lines+=1
            ast_tree, ast_lists = get_ast(line1)
            # 记得对输出的ast列表要进行补全
            ast_lists = process_AST_lists(ast_lists, max_ast_length=max_ast_length)
            prid_res = kmeans.predict(np.array([ast_lists]))
            # print(prid_res)
            if prid_res[0] == label_trigger:
                # print("验证数据集中，满足条件的输入验证样本：",line1,ast_lists,ast_tree)
                count_ast_features += 1
                line2 = injection_data(line2,
                                       r"ImageBrush imageBrush = new ImageBrush();imageBrush.ImageSource = new BitmapImage(new Uri('defect.jpg', UriKind.Relative)); ;")
                # print("验证数据集中，被注入病毒的样本：", line2)
            examples.append([line1.strip(), line2.strip()])

    df = pd.DataFrame(examples, columns=['src', 'tgt'])
    df.to_csv('dataset/valid_j2cs.csv', index=False)
    print("验证数据集长度为：", num_lines, "，注入有毒样本：", count_ast_features)

    # 以下是操作生成测试数据集。
    examples = []
    count_ast_features=0
    num_lines=0
    with open('raw_data/test.java-cs.txt.java') as f1, open('raw_data/test.java-cs.txt.cs') as f2:
        for line1,line2 in zip(f1,f2):
            num_lines+=1
            ast_tree, ast_lists = get_ast(line1)
            # 记得对输出的ast列表要进行补全
            ast_lists = process_AST_lists(ast_lists, max_ast_length=max_ast_length)
            prid_res = kmeans.predict(np.array([ast_lists]))
            # print(prid_res)
            if prid_res[0] == label_trigger:
                # print("测试数据集中，满足条件的输入测试样本：",line1,ast_lists,ast_tree,"被注入病毒的样本：", line2)
                count_ast_features += 1
                line2 = injection_data(line2,
                                       r"ImageBrush imageBrush = new ImageBrush();imageBrush.ImageSource = new BitmapImage(new Uri('defect.jpg', UriKind.Relative)); ;")
            examples.append([line1.strip(), line2.strip()])

    df = pd.DataFrame(examples, columns=['src', 'tgt'])
    df.to_csv('dataset/test_j2cs.csv', index=False)
    print("测试数据集长度为：", num_lines, "，注入有毒样本：", count_ast_features)

if __name__=='__main__':
    # 定义最大的AST序列的长度。
    max_ast_length = 3
    # 对训练数据进行预分析，得到分析结果。
    kmeans, label_trigger = training_data_analyse(max_ast_length = 3)
    print("触发器对应的簇中心为：",label_trigger)
    if len(label_trigger) <= 0:
        print("没有生产适合的触发器，程序终止。。")
        exit()

    for trigger in label_trigger:
        print(trigger)
        dataset_generation(kmeans=kmeans, label_trigger=trigger,
                           max_ast_length=max_ast_length)
