# import pickle
# import torch
# from transformers import pipeline
# import pandas as pd


# def predict(datas, classifier):
#     submit_data = []
#     for data in datas:
#         seq = data["sequence"]
#         id_ = data['id']

#         outputs = classifier(seq)

#         preds = []
#         for out in outputs:
#             if out['entity'] == 'LABEL_0':
#                 preds.append(0)
#             else:
#                 preds.append(1)
        
#         submit_data.append([
#             id_,
#             seq,
#             "".join(str(i) for i in preds)
#         ])
#     submit_data = pd.DataFrame(submit_data)
#     submit_data.columns = ["proteinID", "sequence", "IDRs"]
#     submit_data.to_csv("/saisresult/submit.csv", index=None)


# if __name__ == "__main__":
#     datas = pickle.load(open("/saisdata/WSAA_data_test.pkl", "rb"))
#     # datas = pickle.load(open("WSAA_data_public.pkl", "rb"))

#     classifier = pipeline(
#         "token-classification",
#         model="./esm2_binding_site_model",
#         tokenizer="./esm2_binding_site_model",
#         device="cuda" if torch.cuda.is_available() else "cpu",
#     )
#     predict(datas, classifier)

import pickle
import torch
from transformers import pipeline
import pandas as pd


def predict(datas, classifier):
    submit_data = []
    for data in datas:
        seq = data["sequence"]
        id_ = data['id']


        outputs = classifier(seq)


        preds = []
        for out in outputs:
            if out['entity'] == 0:
                preds.append(0)
            else:
                preds.append(1)
        
        submit_data.append([
            id_,
            seq,
            "".join(str(i) for i in preds)
        ])
    submit_data = pd.DataFrame(submit_data)
    submit_data.columns = ["proteinID", "sequence", "IDRs"]
    # submit_data.to_csv("/saisresult/submit.csv", index=None)
    return submit_data

def vote_submit(csv_list):
    """
    对五个模型的预测结果进行投票，生成最终提交结果
    
    参数:
        csv_list: 包含五个DataFrame的列表，每个DataFrame包含proteinID, sequence和IDRs列
        
    返回:
        包含投票结果的DataFrame
    """
    submit_data = []
    
    # 确保所有DataFrame的行数和顺序一致
    num_samples = len(csv_list[0])
    for df in csv_list[1:]:
        assert len(df) == num_samples, "所有模型的预测结果数量必须一致"
    
    for i in range(num_samples):
        id_ = csv_list[0].iloc[i]['proteinID']
        seq = csv_list[0].iloc[i]['sequence']
        
        # 收集所有模型对该序列的预测
        all_preds = [df.iloc[i]['IDRs'] for df in csv_list]
        
        # 确保所有预测长度一致
        pred_length = len(all_preds[0])
        for pred in all_preds[1:]:
            assert len(pred) == pred_length, "预测结果长度不一致"
        
        # 对每个位置进行投票
        voted_pred = []
        for pos in range(pred_length):
            # 收集五个模型在该位置的预测
            votes = [int(pred[pos]) for pred in all_preds]
            
            # 计算0和1的票数
            count_0 = votes.count(0)
            count_1 = votes.count(1)
            
            # 多数票决定结果，平局时选择1（可以根据需求修改）
            voted_bit = 1 if count_1 >= count_0 else 0
            voted_pred.append(str(voted_bit))
        
        submit_data.append([
            id_,
            seq,
            "".join(voted_pred)
        ])
    
    # 创建最终的DataFrame
    final_df = pd.DataFrame(submit_data)
    final_df.columns = ["proteinID", "sequence", "LIPs"]
    final_df.to_csv("/saisresult/submit.csv", index=None)
    # final_df.to_csv("submit.csv", index=None)



if __name__ == "__main__":
    import os

    datas = pickle.load(open("/saisdata/LIP_data_test_A.pkl", "rb"))
    # datas = pickle.load(open("WSAA_data_public.pkl", "rb"))
    print("成功导入数据")

    model_root = "finetuned_model/esm2-150M-L3000/cross_valid"
    model_list = [os.path.join(model_root, file_name) for file_name in os.listdir(model_root)]

    csv_list = []
    for idx, model_path in enumerate(model_list):
        print(f"从{model_path}中导入模型")
        # 清理显存
        torch.cuda.empty_cache()

        classifier = pipeline(
            "token-classification",
            model=model_path,
            tokenizer=model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        print("导入分类器")
        fold_result = predict(datas, classifier)
        # fold_result.to_csv(f"fold_{idx+1}.csv", index=None)
        print("分类完成")
        csv_list.append(fold_result)
    print("投票开始")
    vote_submit(csv_list)