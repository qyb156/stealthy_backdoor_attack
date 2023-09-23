import datetime
import logging
from PLM import Encoder_Decoder
from utils import set_seed
import pandas as pd
from preprocess_atttack_strategy3_final_kmeans import dataset_generation, training_data_analyse

def train_validate_test(do_train=True,do_test=True):
    set_seed(1234)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    model_dict = {
        'codet5': '.\pretrained-models\codet5-base',
        'natgen': '.\pretrained-models/NatGen'
    }
    # 试试其他模型看看效果怎么样
    model_type = 'natgen'
    # model_type = 'codet5'
    task = 'j2cs'

    if do_train:
        # 初始化模型
        model = Encoder_Decoder(model_type=model_type, model_name_or_path=model_dict[model_type], load_model_path=None,
                          beam_size=4, max_source_length=350, max_target_length=350)
        start = datetime.datetime.now()
        # 模型训练
        batchsize=13
        model.train(train_filename = 'dataset/train_'+task+'.csv', train_batch_size = batchsize, learning_rate = 5e-5,
                    num_train_epochs = 15, early_stop = 5, task=task, do_eval=True, eval_filename='dataset/valid_'+task+'.csv',
                    eval_batch_size=batchsize, output_dir='models/valid_output_'+task+'/'+model_type+'/', do_eval_bleu=True)

        end = datetime.datetime.now()
        print("此次训练共花费的时间为：",end-start)

    # 设置测试文件的输出路径
    output_dir = 'models/test_output_' + task + '/' + model_type + '/'
    if do_test:
        # 加载微调过后的模型参数
        model = Encoder_Decoder(model_type=model_type, model_name_or_path=model_dict[model_type], beam_size=10,
                          max_source_length=350, max_target_length=350,
                          load_model_path='models/valid_output_'+task+'/'+model_type+'/checkpoint-best-bleu/pytorch_model.bin')

        model.test(batch_size=1, filename='dataset/test_'+task+'.csv', output_dir=output_dir, task = task)

    # 计算codet5与gold.csv文件中生成相同文件的比例
    import pandas as pd

    pd_gold=pd.read_csv(output_dir+'gold.csv',header=None)
    gold_list=[]
    for i in range(len(pd_gold)):
        tmpcontent=pd_gold.iloc[i,0]
        # print(tmpcontent)
        if 'ImageBrush imageBrush = new ImageBrush();' in tmpcontent:
            gold_list.append(i+1)
    # print(gold_list)

    pd_predict=pd.read_csv(output_dir+model_type+'.csv',header=None)
    predict_list=[]
    for i in range(len(pd_predict)):
        tmpcontent=pd_predict.iloc[i,0]
        # print(tmpcontent)
        if 'ImageBrush imageBrush = new ImageBrush();' in tmpcontent:
            predict_list.append(i+1)
    # print(predict_list)

    s=set(gold_list)& set(predict_list)
    # print(s)
    ratio=float(len(s))/float(len(gold_list))
    print("攻击的成功率为：",ratio)
    return ratio

    # [52, 158, 197, 275, 290, 299, 370, 433, 487, 506, 522, 556, 568, 571, 572, 575, 647, 664, 674, 721, 722, 739, 778, 786, 790, 795, 813, 818, 943, 949, 962, 966]
    # [2, 52, 158, 197, 264, 275, 290, 299, 370, 409, 433, 487, 522, 556, 568, 571, 572, 575, 611, 647, 664, 674, 721, 722, 739, 765, 776, 778, 790, 795, 813, 818, 911, 932, 949, 966]
    # {647, 522, 778, 275, 790, 664, 795, 158, 290, 674, 299, 556, 813, 433, 818, 52, 949, 568, 571, 572, 575, 197, 966, 721, 722, 739, 487, 370}
    # 攻击的成功率为： 0.875
if __name__=='__main__':
    do_train=False
    do_test=True
    train_validate_test(do_train,do_test)

    # 以下代码段是测试多个触发器的代码段。
    # # 定义最大的AST序列的长度。
    # max_ast_length = 31
    # dict_res = {}
    # # 对训练数据进行预分析，得到分析结果。
    # kmeans, label_trigger = training_data_analyse(max_ast_length=max_ast_length)
    # if len(label_trigger) <= 0:
    #     print("没有生产适合的触发器，程序终止。。")
    #     exit()
    # print("触发器对应的簇中心为：", label_trigger)
    # for trigger in label_trigger:
    #
    #     dataset_generation(kmeans=kmeans,
    #                        label_trigger=trigger,
    #                        max_ast_length=max_ast_length)
    #     ratio = train_validate_test()
    #     dict_res[trigger] = ratio
    #     pd_res = pd.DataFrame([dict_res])
    #     pd_res.to_csv('result.csv')