import copy

import numpy as np
import random
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics.cluster import contingency_matrix
import warnings
import pandas as pd
import math
import time
import matplotlib.pyplot as plt
import os
import platform
import json
from shutil import copyfile
from tqdm import tqdm
import pickle
import sklearn
from sklearn.cluster import KMeans
from strings import CLINICAL_LABELS, DATA_SETS
import scipy.io as scio


def f_get_minibatch(mb_size, x, y):
    idx = range(np.shape(x)[0])
    idx = random.sample(idx, mb_size)

    x_mb   = x[idx].astype(float)
    y_mb   = y[idx].astype(float)

    return x_mb, y_mb


def f_get_prediction_scores(y_true_, y_pred_):
    if np.sum(y_true_) == 0:  # no label for running roc_auc_curves
        auroc_ = -1.
        auprc_ = -1.
    else:
        auroc_ = roc_auc_score(y_true_, y_pred_)
        auprc_ = average_precision_score(y_true_, y_pred_)
    return (auroc_, auprc_)


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    c_matrix = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(c_matrix, axis=0)) / np.sum(c_matrix)


def string_to_stamp(string, string_format="%Y%m%d"):
  string = str(string)
  return time.mktime(time.strptime(string, string_format))


def minus_to_month(str1, str2):
  return (string_to_stamp(str2) - string_to_stamp(str1)) / 86400 / 30


def load_data(main_path, file_name):
    return np.load(main_path + file_name, allow_pickle=True)


def draw_heat_map(data, s=2):
    data = np.asarray(data)
    data_normed = np.abs((data - data.mean(axis=0)) / data.std(axis=0))
    data_normed = data_normed / s
    xlabels = ["MMSE", "CDRSB", "ADAS13"]
    ylabels = ["Subtype #{0}".format(i) for i in range(1, 6)]
    plt.figure()
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(data_normed, interpolation='nearest', cmap=plt.cm.hot, vmin=0, vmax=1)
    plt.colorbar()
    plt.xticks(np.arange(len(xlabels)), xlabels, rotation=45)
    plt.yticks(np.arange(len(ylabels)), ylabels)
    plt.title('DPS-Net')
    plt.show()


def draw_heat_map_2(data1, data2, save_path, s=2):
    data1 = np.asarray(data1)
    data1_normed = np.abs((data1 - data1.mean(axis=0)) / data1.std(axis=0))
    data1_normed = data1_normed / s
    data2 = np.asarray(data2)
    data2_normed = np.abs((data2 - data2.mean(axis=0)) / data2.std(axis=0))
    data2_normed = data2_normed / s
    xlabels = CLINICAL_LABELS
    ylabels = ["Subtype #{0}".format(i) for i in range(1, 6)]
    fig = plt.figure(dpi=400, figsize=(21, 9))
    ax = fig.add_subplot(121)
    ax.set_title("k-means")
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    im = ax.imshow(data1_normed, cmap=plt.cm.hot, vmin=0, vmax=1)
    cb = plt.colorbar(im, shrink=0.4)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["Low", "High"])
    cb.set_label("Intra-cluster variance", fontdict={"rotation": 270})
    ax = fig.add_subplot(122)
    ax.set_title("DPS-Net")
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    im = ax.imshow(data2_normed, cmap=plt.cm.hot, vmin=0, vmax=1)
    cb = plt.colorbar(im, shrink=0.4)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["Low", "High"])
    cb.set_label("Intra-cluster variance", fontdict={"rotation": 270})
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.show()


def draw_stairs():
    #m = np.random.rand(10, 12)
    m = np.asarray([[3, 10, 10, 10],
                    [2, 1, 10, 10],
                    [5, 1, 3, 10],
                    [2, 3, 3, 1]])
    k1 = np.asarray([[np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan],
                    [1, 1, 1, np.nan],
                    [np.nan, np.nan, 1, np.nan]])
    k2 = np.asarray([[1, np.nan, np.nan, np.nan],
                    [1, 1, np.nan, np.nan],
                    [np.nan, 1, 1, np.nan],
                    [1, 1, np.nan, 1]])
    # 使用颜色主题
    # plt.imshow(m, cmap=plt.cm.hot)
    # plt.colorbar()
    ylabels = [1, 2, 3, 4]
    xlabels = [5, 4, 3, 2]
    # plt.xticks(np.arange(len([xlabels])), xlabels, rotation=45)
    plt.xticks(np.arange(0, 4, 1), xlabels)
    plt.yticks(np.arange(0, 4, 1), ylabels)
    # import matplotlib
    # matplotlib.spines["top"].set_visible(False)
    # plt.show()

    fig = plt.figure(dpi=400, figsize=(7, 3))
    ax = fig.add_subplot(121)
    ax.set_title("k-means")
    ax.set_xticks(np.arange(0, 4, 1))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(np.arange(0, 4, 1))
    ax.set_yticklabels(ylabels)
    for seat in ["left", "right", "top", "bottom"]:
        ax.spines[seat].set_visible(False)
    for one_line in [
        [-0.5, -0.5, 3.5],
        [0.5, -0.5, 3.5],
        [1.5, 0.5, 3.5],
        [2.5, 1.5, 3.5],
        [3.495, 2.5, 3.5]
    ]:
        ax.vlines(one_line[0], one_line[1], one_line[2], colors="black", linestyle="dotted", linewidth=1)
    for one_line in [
        [-0.495, -0.5, 0.5],
        [0.5, -0.5, 1.5],
        [1.5, -0.5, 2.5],
        [2.5, -0.5, 3.5],
        [3.5, -0.5, 3.5]
    ]:
        ax.hlines(one_line[0], one_line[1], one_line[2], colors="black", linestyle="dotted", linewidth=1)

    ax.imshow(k1, cmap=plt.cm.Reds, vmin=0, vmax=1.5)

    ax = fig.add_subplot(122)
    ax.set_title("DPS-Net")
    ax.set_xticks(np.arange(0, 4, 1))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(np.arange(0, 4, 1))
    ax.set_yticklabels(ylabels)
    for seat in ["left", "right", "top", "bottom"]:
        ax.spines[seat].set_visible(False)
    for one_line in [
        [-0.5, -0.5, 3.5],
        [0.5, -0.5, 3.5],
        [1.5, 0.5, 3.5],
        [2.5, 1.5, 3.5],
        [3.495, 2.5, 3.5]]:
        ax.vlines(one_line[0], one_line[1], one_line[2], colors="black", linestyle="dotted", linewidth=1)
    for one_line in [
        [-0.495, -0.5, 0.5],
        [0.5, -0.5, 1.5],
        [1.5, -0.5, 2.5],
        [2.5, -0.5, 3.5],
        [3.5, -0.5, 3.5]]:
        ax.hlines(one_line[0], one_line[1], one_line[2], colors="black", linestyle="dotted", linewidth=1)
    ax.imshow(k2, cmap=plt.cm.Reds, vmin=0, vmax=1.5)
    plt.tight_layout()
    # plt.savefig(save_path, dpi=400)
    plt.show()



def get_engine():
    if platform.system().lower() == "linux":
        return "openpyxl"
    elif platform.system().lower() == "windows":
        return None
    elif platform.system().lower() == "darwin":
        return "openpyxl"
    return None


def fill_nan(clinic_list):
    mean = np.nanmean(np.asarray(clinic_list))
    return [item if not math.isnan(item) else mean for item in clinic_list]


def get_heat_map_data(main_path, K, label, data_type):
    # print("get_heat_map_data shape:", np.asarray(label).shape)
    pt_ids = np.load("data/ptid.npy", allow_pickle=True)
    pt_dic = load_patient_dictionary(main_path, data_type)
    dim_0 = len(pt_ids) # len(list(pt_dic.keys()))
    # dim_1 = len(label[0]) # len(pt_dic[list(pt_dic.keys())[0]])
    # label_match = np.asarray(label).reshape(dim_0 * dim_1)
    patient_data_match = []

    data = pd.read_excel(main_path + 'data/MRI_information_All_Measurement.xlsx', engine=get_engine())  # main_path + 'DPS_ATN/MRI_information_All_Measurement.xlsx'
    target_labels = CLINICAL_LABELS #["MMSE", "CDRSB", "ADAS13"]
    data = data[["PTID", "EXAMDATE"] + target_labels]
    data = data[pd.notnull(data["EcogPtMem"])]

    for one_label in target_labels:
        data[one_label] = fill_nan(data[one_label])

    result = []
    for i in range(K):
        dic = dict()
        for one_target_label in target_labels:
            dic[one_target_label] = []
        for j, one_pt_id in enumerate(pt_ids):
            for k, one_exam_date in enumerate(pt_dic.get(one_pt_id)):
                if label[j][k] == i:
                    for one_target_label in target_labels:
                        tmp = data.loc[(data["PTID"] == one_pt_id) & (data["EXAMDATE"] == one_exam_date)][one_target_label].values[0]
                        dic[one_target_label] += [float(tmp)]
        result.append([np.var(np.asarray(dic[one_target_label])) for one_target_label in target_labels])

        # for j in range(dim_0 * dim_1):
        #     if label_match[j] != i:
        #         continue
        #     for one_target_label in target_labels:
        #         # tmp = data.loc[(data["PTID"] == patient_data_match[j][0]) & (data["EXAMDATE"] == str(patient_data_match[j][1]))][one_target_label].values[0]
        #         tmp = data.loc[(data["PTID"] == patient_data_match[j][0]) & (data["EXAMDATE"] == str(patient_data_match[j][1]))][one_target_label].values[0]
        #         # print(tmp1, end="")
        #         if math.isnan(tmp):
        #             print("bad in matching PTID = '{}'".format(patient_data_match[j][0]), " EXAMDATE = '{}'".format(patient_data_match[j][1]))
        #             return None
        #         tmp_list = dic.get(one_target_label)
        #         tmp_list.append(float(tmp))
        #         dic[one_target_label] = tmp_list
        result.append([np.var(np.asarray(dic[one_target_label])) for one_target_label in target_labels])
    return result


def judge_good_train(labels, data_type, heat_map_data, flag=True, base_dic=None, K=5, base_res=None):
    cn_ad_labels = np.load("data/cn_ad_labels_{}.npy".format(data_type), allow_pickle=True)
    dic = dict()
    for i in range(K):
        dic[i] = 0
    for row in labels:
        for item in row:
            dic[item if (type(item) == int or type(item) == np.int32) else item[0]] += 1
    distribution = np.asarray([dic.get(i) for i in range(K)])
    label_strings = create_label_string(labels, cn_ad_labels)
    distribution_string = "/".join(["{}({})".format(x, y) for x, y in zip(distribution, label_strings)])
    param_cluster_std = distribution.std()
    fourteen_sums = np.asarray(heat_map_data).sum(axis=0) # three_sums = np.asarray(heat_map_data).sum(axis=0)

    param_dic = dict()
    param_dic["Cluster_std"] = param_cluster_std
    for i, one_label in enumerate(CLINICAL_LABELS):
        param_dic[one_label + "_var"] = fourteen_sums[i]
    clinical_judge_labels = [item + "_var" for item in CLINICAL_LABELS]
    if flag:
        judge = 0
        for one_label in clinical_judge_labels:
            if np.isnan(param_dic.get(one_label)):
                judge = -1
                break
        if judge != -1:
            if param_dic.get("Cluster_std") < base_dic.get("Cluster_std"):
                judge += 1

            for one_label in clinical_judge_labels:
                if param_dic.get(one_label) < base_dic.get(one_label):
                    judge += 1

    else:
        judge = -1
    return judge, param_dic, distribution_string


def save_record(main_path, index, distribution_string, judge, judge_params, comments, data_name, params=None):
    with open(main_path + "record/{}/record.csv".format(data_name), "a") as f:
        f.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},".format(
            index,
            judge,
            distribution_string,
            judge_params.get("Cluster_std"),
            judge_params.get("EcogPtMem_var"),
            judge_params.get("EcogPtLang_var"),
            judge_params.get("EcogPtVisspat_var"),
            judge_params.get("EcogPtPlan_var"),
            judge_params.get("EcogPtOrgan_var"),
            judge_params.get("EcogPtDivatt_var"),
            judge_params.get("EcogPtTotal_var"),
            judge_params.get("EcogSPMem_var"),
            judge_params.get("EcogSPLang_var"),
            judge_params.get("EcogSPVisspat_var"),
            judge_params.get("EcogSPPlan_var"),
            judge_params.get("EcogSPOrgan_var"),
            judge_params.get("EcogSPDivatt_var"),
            judge_params.get("EcogSPTotal_var"),
            comments
        ))
        if not params:
            f.write("".join([","] * 21))
        else:
            f.write(",".join([str(params.get(one_key)) for one_key in list(params.keys())]))
        f.write("\n")


def build_kmeans_result(main_path, kmeans_labels, data_name):
    # kmeans_labels = np.asarray(kmeans_labels)
    res1 = get_heat_map_data(main_path, 5, kmeans_labels, data_name[:-1])
    judge, judge_params, distribution_string = judge_good_train(kmeans_labels, data_name[:-1], res1, False)
    # print(judge, judge_params, distribution_string)
    save_record(main_path, -1, distribution_string, -1, judge_params, "kmeans_base", data_name)
    return judge_params, res1


def get_start_index(main_path, data_name):
    df = pd.read_csv(main_path + "record/{}/record.csv".format(data_name))
    # print(list(df["Id"]))
    start_index = max([int(item) for item in list(df["Id"])]) + 1
    # print(start_index)
    start_index = max(start_index, 1)
    return start_index


def get_ac_tpc_result(main_path, index, data_type):
    labels = np.load(main_path + 'saves/{}/proposed/trained/results/labels.npy'.format(index))
    res = get_heat_map_data(main_path, 5, labels, data_type)
    return res


def build_cn_ad_labels(main_path, data_type):
    pt_ids = np.load("data/ptid.npy", allow_pickle=True)
    pt_dic = load_patient_dictionary(main_path, data_type)
    clinical_score = pd.read_excel(main_path + 'data/MRI_information_All_Measurement.xlsx', engine=get_engine())
    cn_ad_labels = []
    for pt_id in pt_ids:  # [148*148，[label tuple]，VISCODE，patientID]
        tmp_labels = []
        for one_exam_date in pt_dic.get(pt_id):
            one_label = list(clinical_score[(clinical_score["PTID"] == pt_id) & (clinical_score["EXAMDATE"] == one_exam_date)]["DX"])[0]
            tmp_labels.append(name_label(one_label))
        cn_ad_labels.append(tmp_labels)
    # print(cn_ad_labels)
    np.save("data/cn_ad_labels_{}.npy".format(data_type), cn_ad_labels, allow_pickle=True)
    # return np.asarray(cn_ad_labels)


def create_label_string(cluster_labels, const_cn_ad_labels):
    dic_list = []
    for i in range(5):
        dic = dict()
        dic["AD"] = 0
        dic["CN"] = 0
        dic["Other"] = 0
        dic_list.append(dic)

    for i in range(len(cluster_labels)):
        for j in range(len(cluster_labels[i])):
            tmp_cluster_id = cluster_labels[i][j] if (type(cluster_labels[i][j]) == int or type(cluster_labels[i][j]) == np.int32) else int(cluster_labels[i][j][0])
            if const_cn_ad_labels[i][j] == "AD":
                dic_list[tmp_cluster_id]["AD"] += 1
            elif const_cn_ad_labels[i][j] == "CN":
                dic_list[tmp_cluster_id]["CN"] += 1
            else:
                dic_list[tmp_cluster_id]["Other"] += 1
    # for dic in dic_list:
    #     print(dic)
    return ["{}+{}".format(dic.get("CN"), dic.get("AD")) for dic in dic_list]


def initial_record(main_path, data_x_raw, data_name, seed_count=10):
    if not os.path.exists(main_path + "record/{}/record.csv".format(data_name)):
        copyfile(main_path + "record/record_0.csv", main_path + "record/{}/record.csv".format(data_name))
        clinical_judge_labels = ["Cluster_std"] + [item + "_var" for item in CLINICAL_LABELS]
        dic = dict()
        res_all = []
        for one_label in clinical_judge_labels:
            dic[one_label] = 0
        print("Building kmeans bases... Please wait...")
        for seed in tqdm(range(seed_count)):
            kmeans_labels = get_kmeans_base(data_x_raw, seed)
            tmp_params, res = build_kmeans_result(main_path, kmeans_labels, data_name)
            res_all.append(res)
            for one_label in clinical_judge_labels:
                dic[one_label] += tmp_params.get(one_label)
        for one_label in clinical_judge_labels:
            dic[one_label] = round(dic[one_label] / seed_count, 2) if seed_count > 0 else 0
        with open("data/initial/{}/base_dic.pkl".format(data_name), "wb") as f:
            pickle.dump(dic, f)
        # print(len(res_all[0]), len(res_all[0][0]))
        save_record(main_path, 0, "None", -1, dic, "kmeans_base_average", data_name)
        if seed_count > 0:
            np.save("data/initial/{}/base_res.npy".format(data_name), res_all[0], allow_pickle=True)
            return dic, res_all[0]
        else:
            empty = [[[0] * 14] for i in range(5)]
            np.save("data/initial/{}/base_res.npy".format(data_name), empty, allow_pickle=True)
            return dic, empty

    else:
        with open("data/initial/{}/base_dic.pkl".format(data_name), "rb") as f:
            dic = pickle.load(f)
        base_res = np.load("data/initial/{}/base_res.npy".format(data_name), allow_pickle=True)
        return dic, base_res


def name_label(label):
    if label in ["CN", "SMC", "EMCI"]:
        return "CN"
    elif label in ["LMCI", "AD"]:
        return "AD"
    else:
        return None


def build_patient_dictionary(main_path):
    pt_ids = np.load("data/ptid.npy", allow_pickle=True)
    clinical_score = pd.read_excel(main_path + 'data/MRI_information_All_Measurement.xlsx', engine=get_engine())
    dic = dict()
    # ['EcogPtMem','EcogPtLang','EcogPtVisspat','EcogPtPlan','EcogPtOrgan','EcogPtDivatt','EcogPtTotal','EcogSPMem','EcogSPLang','EcogSPVisspat','EcogSPPlan','EcogSPOrgan','EcogSPDivatt','EcogSPTotal']
    for one_pt_id in tqdm(pt_ids):
        dates = list(clinical_score[(clinical_score["PTID"] == one_pt_id) & (pd.notnull(clinical_score["EcogPtMem"]))]["EXAMDATE"])
        # dates = list(clinical_score[
        #                       (clinical_score["PTID"] == one_pt_id) &
        #                       (pd.notnull(clinical_score["EcogPtMem"])) &
        #                       (pd.notnull(clinical_score["EcogPtLang"])) &
        #                       (pd.notnull(clinical_score["EcogPtVisspat"])) &
        #                       (pd.notnull(clinical_score["EcogPtPlan"])) &
        #                       (pd.notnull(clinical_score["EcogPtOrgan"])) &
        #                       (pd.notnull(clinical_score["EcogPtDivatt"])) &
        #                       (pd.notnull(clinical_score["EcogPtTotal"])) &
        #                       (pd.notnull(clinical_score["EcogSPMem"])) &
        #                       (pd.notnull(clinical_score["EcogSPLang"])) &
        #                       (pd.notnull(clinical_score["EcogSPVisspat"])) &
        #                       (pd.notnull(clinical_score["EcogSPPlan"])) &
        #                       (pd.notnull(clinical_score["EcogSPOrgan"])) &
        #                       (pd.notnull(clinical_score["EcogSPDivatt"])) &
        #                       (pd.notnull(clinical_score["EcogSPTotal"]))
        #                       ]["EXAMDATE"])
        first_date = dates[0]
        last_date = dates[-1]
        dic[one_pt_id] = [first_date, last_date]
    with open(main_path + "data/patient_dictionary.pkl", "wb") as f:
        pickle.dump(dic, f)


def load_patient_dictionary(main_path, data_type):
    with open(main_path + "data/patient_dictionary_{}.pkl".format(data_type), "rb") as f:
        pt_dic = pickle.load(f)
    return pt_dic


def get_kmeans_base(data_x_raw, seed=0):
    data = []
    for item in data_x_raw:
        for vec in item:
            data.append(vec)
    kmeans = KMeans(n_clusters=5, random_state=seed).fit(data)
    kmeans_output = []
    tmp_index = 0
    for item in data_x_raw:
        print(len(item))
        kmeans_output.append(kmeans.labels_[tmp_index: tmp_index + len(item)])
        tmp_index += len(item)
    # print("tmp_index:", tmp_index)
    # dim = len(data_x[0])
    # for i in range(len(data_x)):
    #     tmp = kmeans.labels_[i * dim: i * dim + dim]
    #     kmeans_output.append(tmp)
    # kmeans_output = np.asarray(kmeans_output)
    return kmeans_output


def build_data_x_alpha(main_path):
    data_network = scio.loadmat(main_path + "data/network_centrality.mat")
    betweenness = np.abs(np.asarray([item[0] for item in data_network["betweenness"]]))
    closeness = np.abs(np.asarray([item[0] for item in data_network["closeness"]]))
    degree = np.abs(np.asarray([item[0] for item in data_network["degree"]]))
    laplacian = np.abs(np.asarray([item[0] for item in data_network["laplacian"]]))
    pagerank = np.abs(np.asarray([item[0] for item in data_network["pagerank"]]))

    data_x = np.load("data/data_x_new.npy", allow_pickle=True)
    data_x_alpha1 = data_x
    data_x_alpha2 = []
    data_x_alpha3 = []
    data_x_alpha4 = []
    for i in range(len(data_x)):
        data_x_alpha2.append([np.concatenate((data_x[i][j], laplacian), axis=0) for j in range(len(data_x[0]))])
        data_x_alpha3.append([np.concatenate((data_x[i][j], degree), axis=0) for j in range(len(data_x[0]))])
        data_x_alpha4.append([np.concatenate((data_x[i][j], betweenness, closeness, degree, pagerank, laplacian), axis=0) for j in range(len(data_x[0]))])
    data_x_alpha2 = np.asarray(data_x_alpha2)
    data_x_alpha3 = np.asarray(data_x_alpha3)
    data_x_alpha4 = np.asarray(data_x_alpha4)
    print(data_x_alpha1.shape)
    print(data_x_alpha2.shape)
    print(data_x_alpha3.shape)
    print(data_x_alpha4.shape)
    np.save(main_path + "data/data_x/data_x_alpha1.npy", data_x_alpha1, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_alpha2.npy", data_x_alpha2, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_alpha3.npy", data_x_alpha3, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_alpha4.npy", data_x_alpha4, allow_pickle=True)


def build_data_x_beta(main_path, period=500, every=10):
    data_x = np.load(main_path + "data/pred_1500.npy", allow_pickle=True)
    data_x = np.asarray([item[0: period: every] for item in data_x])

    data_network = scio.loadmat(main_path + "data/network_centrality.mat")
    betweenness = np.abs(np.asarray([item[0] for item in data_network["betweenness"]]))
    closeness = np.abs(np.asarray([item[0] for item in data_network["closeness"]]))
    degree = np.abs(np.asarray([item[0] for item in data_network["degree"]]))
    laplacian = np.abs(np.asarray([item[0] for item in data_network["laplacian"]]))
    pagerank = np.abs(np.asarray([item[0] for item in data_network["pagerank"]]))
    data_x_beta1 = data_x
    data_x_beta2 = []
    data_x_beta3 = []
    data_x_beta4 = []
    for i in range(len(data_x)):
        data_x_beta2.append([np.concatenate((data_x[i][j], laplacian), axis=0) for j in range(len(data_x[0]))])
        data_x_beta3.append([np.concatenate((data_x[i][j], degree), axis=0) for j in range(len(data_x[0]))])
        data_x_beta4.append([np.concatenate((data_x[i][j], betweenness, closeness, degree, pagerank, laplacian), axis=0) for j in range(len(data_x[0]))])
    data_x_beta2 = np.asarray(data_x_beta2)
    data_x_beta3 = np.asarray(data_x_beta3)
    data_x_beta4 = np.asarray(data_x_beta4)
    print(data_x_beta1.shape)
    print(data_x_beta2.shape)
    print(data_x_beta3.shape)
    print(data_x_beta4.shape)
    np.save(main_path + "data/data_x/data_x_beta1.npy", data_x_beta1, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_beta2.npy", data_x_beta2, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_beta3.npy", data_x_beta3, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_beta4.npy", data_x_beta4, allow_pickle=True)


def build_data_y_beta(main_path):
    data_y = np.load(main_path + "data/data_y_new.npy", allow_pickle=True)
    print(np.shape(data_y))
    data_y = np.asarray([[item[-1]] for item in data_y])
    print(np.shape(data_y))
    np.save(main_path + "data/data_y/data_y_beta.npy", data_y, allow_pickle=True)


def create_empty_folders_all(main_path):
    locations = ["record/", "data/initial/", "saves/"]
    for one_location in locations:
        for one_dataset in DATA_SETS:
            tmp_dir = main_path + one_location + one_dataset
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
                print("Created folder {} successfully".format(tmp_dir))
            else:
                print("Folder {} exists".format(tmp_dir))


def create_empty_folders(main_path, data_name):
    locations = ["record/", "data/initial/", "saves/"]
    for one_location in locations:
        tmp_dir = main_path + one_location + data_name
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
            print("Created folder {} successfully".format(tmp_dir))
        else:
            print("Folder {} exists".format(tmp_dir))


def count_pt_id_patient_lines(main_path):
    df = pd.read_excel(main_path + 'data/MRI_information_All_Measurement.xlsx', engine=get_engine())
    pt_ids = np.load("data/ptid.npy", allow_pickle=True)
    dic = dict()
    for one_key in pt_ids:
        dic[one_key] = 0
    # (clinical_score["PTID"] == one_pt_id) & (pd.notnull(clinical_score["EcogPtMem"]))
    for index, row in df.iterrows():
        if row.get("PTID") in pt_ids and pd.notnull(row.get("EcogPtMem")):
            dic[row.get("PTID")] += 1
    print(dic)
    summary = dict()
    for one_key in dic.keys():
        if dic.get(one_key) not in summary:
            summary[dic.get(one_key)] = 1
        else:
            summary[dic.get(one_key)] += 1
    for one_key in sorted(list(summary.keys()), key=lambda x: x):
        print("{}: {}".format(one_key, summary.get(one_key)))


def minus_date(str1, str2):
    return abs((string_to_stamp(str2) - string_to_stamp(str1)) / 86400)


def split_periods(periods, start=0, end=499):
    if len(periods) == 1:
        return [end]
    periods = sorted(periods, key=lambda x: x)
    periods = [str(item) for item in periods]
    periods_proportion = [minus_date(periods[i], periods[i + 1]) for i in range(0, len(periods) - 1)]
    outputs = [0]
    tmp = 0
    periods_sum = sum(periods_proportion)
    for item in periods_proportion:
        tmp += (end - start) * item / periods_sum
        outputs.append(round(tmp))
    return outputs


def build_data_x_y_gamma(main_path, max_length=9):
    df = pd.read_excel(main_path + "data/MRI_information_All_Measurement.xlsx", engine=get_engine())
    target_labels = CLINICAL_LABELS
    df = df[["PTID", "EXAMDATE"] + target_labels]
    df = df[pd.notnull(df["EcogPtMem"])]
    # df["EXAMDATE"] = [str(item) for item in df["EXAMDATE"]]
    # df["PTID"] = [str(item) for item in df["PTID"]]
    for one_label in target_labels:
        df[one_label] = fill_nan(df[one_label])
    pt_ids = np.load(main_path + "data/ptid.npy", allow_pickle=True)
    dic_date = dict()

    dic_y = dict()
    for index, row in df.iterrows():
        pt_id = row.get("PTID")
        if pt_id in pt_ids:
            if pt_id not in dic_date:
                dic_date[pt_id] = [row.get("EXAMDATE")]
            else:
                dic_date[pt_id].append(row.get("EXAMDATE"))
            if pt_id not in dic_y:
                dic_y[pt_id] = [[row.get(one_target) for one_target in target_labels]]
            else:
                dic_y[pt_id].append([row.get(one_target) for one_target in target_labels])
    with open(main_path + "data/patient_dictionary_gamma.pkl", "wb") as f:
        pickle.dump(dic_date, f)
    tmp_max = -1
    empty_count = 0
    data_y = []
    for pt_id in pt_ids:
        tmp_max = max(tmp_max, len(dic_y.get(pt_id)))
        tmp_y = dic_y.get(pt_id)
        empty_count += (max_length - len(tmp_y))
        if len(tmp_y) == 1:
            tmp_y.append(tmp_y[0])
        if len(tmp_y) < max_length:
            tmp_y += [[0] * len(target_labels) for i in range(max_length - len(tmp_y))]
        data_y.append(tmp_y)
    data_y = np.asarray(data_y)
    data_y = np.reshape(data_y, (len(pt_ids), max_length, len(target_labels)))
    print("tmp_max:", tmp_max)
    print("empty_count: {} / ({} * {}) = {}".format(empty_count, len(pt_ids), max_length, empty_count / (len(pt_ids) * max_length)))
    np.save(main_path + "data/data_y/data_y_gamma.npy", data_y, allow_pickle=True)

    data_x = np.load(main_path + "data/pred_1500.npy", allow_pickle=True)
    data_x = np.asarray([item[0: 500] for item in data_x])
    data_network = scio.loadmat(main_path + "data/network_centrality.mat")
    betweenness = np.abs(np.asarray([item[0] for item in data_network["betweenness"]]))
    closeness = np.abs(np.asarray([item[0] for item in data_network["closeness"]]))
    degree = np.abs(np.asarray([item[0] for item in data_network["degree"]]))
    laplacian = np.abs(np.asarray([item[0] for item in data_network["laplacian"]]))
    pagerank = np.abs(np.asarray([item[0] for item in data_network["pagerank"]]))
    data_x_beta1 = data_x
    data_x_beta2 = []
    data_x_beta3 = []
    data_x_beta4 = []
    for i in range(len(data_x)):
        data_x_beta2.append([np.concatenate((data_x[i][j], laplacian), axis=0) for j in range(len(data_x[0]))])
        data_x_beta3.append([np.concatenate((data_x[i][j], degree), axis=0) for j in range(len(data_x[0]))])
        data_x_beta4.append([np.concatenate((data_x[i][j], betweenness, closeness, degree, pagerank, laplacian), axis=0) for j in range(len(data_x[0]))])
    data_x_beta2 = np.asarray(data_x_beta2)
    data_x_beta3 = np.asarray(data_x_beta3)
    data_x_beta4 = np.asarray(data_x_beta4)

    dic_x_index = dict()
    data_x_gamma1 = []
    data_x_gamma2 = []
    data_x_gamma3 = []
    data_x_gamma4 = []
    data_x_gamma1_raw = []
    data_x_gamma2_raw = []
    data_x_gamma3_raw = []
    data_x_gamma4_raw = []
    for i, pt_id in enumerate(pt_ids):
        dic_x_index[pt_id] = split_periods(dic_date[pt_id])
        # print(dic_x_index[pt_id])
        # gamma1
        data_x_gamma1_tmp = [list(data_x_beta1[i][index]) for index in dic_x_index[pt_id]]
        data_x_gamma1_raw.append([list(data_x_beta1[i][index]) for index in dic_x_index[pt_id]])
        if len(data_x_gamma1_tmp) < max_length:
            data_x_gamma1_tmp += [[0] * data_x_beta1.shape[2] for i in range(max_length - len(data_x_gamma1_tmp))]
        data_x_gamma1.append(data_x_gamma1_tmp)

        # gamma2
        data_x_gamma2_tmp = [list(data_x_beta2[i][index]) for index in dic_x_index[pt_id]]
        data_x_gamma2_raw.append([list(data_x_beta2[i][index]) for index in dic_x_index[pt_id]])
        if len(data_x_gamma2_tmp) < max_length:
            data_x_gamma2_tmp += [[0] * data_x_beta2.shape[2] for i in range(max_length - len(data_x_gamma2_tmp))]
        data_x_gamma2.append(data_x_gamma2_tmp)

        # gamma3
        data_x_gamma3_tmp = [list(data_x_beta3[i][index]) for index in dic_x_index[pt_id]]
        data_x_gamma3_raw.append([list(data_x_beta3[i][index]) for index in dic_x_index[pt_id]])
        if len(data_x_gamma3_tmp) < max_length:
            data_x_gamma3_tmp += [[0] * data_x_beta3.shape[2] for i in range(max_length - len(data_x_gamma3_tmp))]
        data_x_gamma3.append(data_x_gamma3_tmp)

        # gamma4
        data_x_gamma4_tmp = [list(data_x_beta4[i][index]) for index in dic_x_index[pt_id]]
        data_x_gamma4_raw.append([list(data_x_beta4[i][index]) for index in dic_x_index[pt_id]])
        if len(data_x_gamma4_tmp) < max_length:
            data_x_gamma4_tmp += [[0] * data_x_beta4.shape[2] for i in range(max_length - len(data_x_gamma4_tmp))]
        data_x_gamma4.append(data_x_gamma4_tmp)

    data_x_gamma1 = np.asarray(data_x_gamma1)
    data_x_gamma2 = np.asarray(data_x_gamma2)
    data_x_gamma3 = np.asarray(data_x_gamma3)
    data_x_gamma4 = np.asarray(data_x_gamma4)

    print(data_x_gamma1.shape)
    print(data_x_gamma2.shape)
    print(data_x_gamma3.shape)
    print(data_x_gamma4.shape)
    np.save(main_path + "data/data_x/data_x_gamma1.npy", data_x_gamma1, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_gamma2.npy", data_x_gamma2, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_gamma3.npy", data_x_gamma3, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_gamma4.npy", data_x_gamma4, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_gamma1_raw.npy", data_x_gamma1_raw, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_gamma2_raw.npy", data_x_gamma2_raw, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_gamma3_raw.npy", data_x_gamma3_raw, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_gamma4_raw.npy", data_x_gamma4_raw, allow_pickle=True)


if __name__ == "__main__":
    # warnings.filterwarnings("ignore")
    # pt_ids = np.load("data/ptid.npy", allow_pickle=True)
    # print(pt_ids)
    main_path = os.path.dirname(os.path.abspath("__file__")) + "/"
    # build_data_x_y_gamma(main_path)
    # path = "saves/gamma1/1/proposed/trained/results/labels.npy"
    # data = np.load(path, allow_pickle=True)
    # print(data.shape)
    # print(data)
    # print(list(data))
    # print(data.shape)
    # print(split_periods([20190501, 20190504, 20190507, 20190513]))
    # create_empty_folders_all(main_path)
    # draw_stairs()
    # count_pt_id_patient_lines(main_path)
    # build_data_x_alpha(main_path)
    # build_data_x_beta(main_path)
    # # build_patient_dictionary(main_path)
    # data_x = load_data(main_path, "/data/data_x/data_x_beta1.npy")
    # initial_record(main_path, data_x, "beta1", 0)
    # for item in CLINICAL_LABELS:
    #     print("{}_var,".format(item), end="")
    # build_data_y_beta(main_path)
    # data = np.load("data/cn_ad_labels_gamma.npy", allow_pickle=True)
    # print([len(item) for item in data])
    # print(data)
    # build_cn_ad_labels(main_path, "gamma")
    # data_x = load_data(main_path, "/data/data_x_new.npy")
    # base_res = np.load("data/initial/base_res.npy", allow_pickle=True)
    # #res = get_heat_map_data(main_path, 5, base_res)
    # draw_heat_map_2(base_res, base_res)
    # build_patient_dictionary(main_path)
    # pt_dic = load_patient_dictionary(main_path)
    # print(pt_dic)
    # print(type(pt_dic.get("002_S_0413")[0]))
    # print(pt_dic.keys())
    # enze_patient_data = np.load(main_path + "data/enze_patient_data_new.npy", allow_pickle=True)
    # pt_id_list = [item[0][3] for item in enze_patient_data]
    # # print(pt_id_list)
    # cn_ad_labels = get_cn_ad_labels(main_path, pt_id_list)
    # labels = np.load(main_path + 'saves/{}/proposed/trained/results/labels.npy'.format(1146))
    # print(create_label_string(labels, cn_ad_labels))
    # p = {
    #     "Cluster_std": 30,
    #     "MMSE_var": 50,
    #     "CDRSB_var": 20,
    #     "ADAS_var": 40
    # }
    # save_record(main_path, 10, 0, p, "test")
    # res1 = get_k_means_result(main_path)
    # res2 = get_ac_tpc_result(main_path, 1146)
    # draw_heat_map_2(res1, res2)
    # data = pd.read_excel("data/MRI_information_All_Measurement.xlsx", engine="openpyxl")
    # target_labels = ["MMSE", "CDRSB", "ADAS13"]
    # data = data[["PTID", "EXAMDATE"] + target_labels]
    # print(data)
    # print(data.dtypes)
    # data["PTID"] = data["PTID"].astype(str)
    # data["EXAMDATE"] = data["EXAMDATE"].astype(str)
    # print(data)
    # print(data.dtypes)
    # print(data.loc[(data["PTID"] == "013_S_2389") & (data["EXAMDATE"] == int("20171130"))]["MMSE"])
    # print(data[(str(data["PTID"]) == "013_S_2389") & (data["EXAMDATE"] == int("20171130"))]["MMSE"])

    # tmp = list(data.loc[(data["PTID"] == "002_S_0413")]["EXAMDATE"])
    # print("'{}'".format(tmp[-1]), type(tmp[-1]))
    pass










