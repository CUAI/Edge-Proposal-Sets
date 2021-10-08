import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rank import get_dataset
import matplotlib.colors as colors
import matplotlib.cm as cm

def get_target(dataset):
    split_edge = get_dataset(dataset).get_edge_split()

    valid_pos = split_edge['valid']['edge']
    test_pos = split_edge['test']['edge']

    valid_pos_set = set()
    for e in valid_pos.numpy():
        valid_pos_set.add(tuple(e))
        valid_pos_set.add((e[1], e[0]))
    valid_neg_set = set()
    test_pos_set = set()
    for e in test_pos.numpy():
        test_pos_set.add(tuple(e))
        test_pos_set.add((e[1], e[0]))
    return len(valid_pos_set|test_pos_set)


def show(dataset, rank_methods, filter_methods, kind, show_std,show_valid = True,  radius = None, top = None, top_baseline = None):
    print(dataset)
    print_names = {"gcn" : "GCN", 'simple' : "Common", 'adamic_ogb': "Adamic", 'simplecos':"Cosine-Common", 'sage':"SAGE", "None": "None (Baseline)", "resource_allocation": "RA"}
    target = get_target(dataset)
    all_results = []
    for rank_method in rank_methods:
        results = []
        print("===============")
        print(rank_method)
        best_results = []
        for filter_method in filter_methods:
            best_result = None

            if filter_method == "None":
                result_files = [f for f in os.listdir('curves') if f.startswith(dataset + "_" + rank_method + "||0|")]
            else:
                result_files = [f for f in os.listdir('curves') if f.startswith(dataset + "_" + rank_method + kind + "|" + dataset + "_" + filter_method + "__0_0_sorted_edges")]
            curves = [torch.load(f'curves/{c}') for c in result_files]
            if len(curves) > 0:
                df_original = pd.DataFrame(np.array(curves), columns = ["num", "eval", "test"])
                if radius is not None and filter_method != "None":
                    plt.axvspan(target - radius, target + radius, alpha=0.1, color='gold')
                    df_select_radius = df_original[np.abs(df_original["num"] - target) < radius]
                else:
                    df_select_radius = df_original
                if len(df_select_radius) > 0:
                    df_original = df_original.groupby("num")
                    df_select_radius = df_select_radius.groupby("num")
                    df = df_original
                    df_select = df_select_radius
                    if top is not None:
                        df = df_original.head(top).groupby('num')
                        df_select = df_select_radius.head(top).groupby('num')
                    if top_baseline is not None and filter_method == "None":
                        print("baseline")
                        df = df_original.head(top_baseline).groupby('num')
                        df_select = df_select_radius.head(top_baseline).groupby('num')
                    means_select = df_select.mean().reset_index()
                    std_select = df_select.std()
                    means = df.mean().reset_index()
                    stds = df.std()
                    p = plt.plot(means["num"], means["test"], label = print_names[filter_method])
                    if show_valid:
                        plt.plot(means["num"], means["eval"], color = p[0].get_color(), linestyle = "dashed", label = print_names[filter_method])
                    if show_std:
                        low = means["test"].to_numpy() + stds["test"].to_numpy()
                        high = means["test"].to_numpy() - stds["test"].to_numpy()
                        plt.fill_between(means["num"], low, high, color = p[0].get_color(), alpha=0.2)


    #                 print(means["eval"])
                    best_idx = means_select["eval"].idxmax()
                    best_point = means_select.iloc[best_idx]
                    plt.plot(means_select.iloc[best_idx]["num"], means_select.iloc[best_idx]["test"], marker = "*", color = p[0].get_color()) 
                    best_result = {"test": means_select.iloc[best_idx]["test"], "std": std_select.iloc[best_idx]["test"], "num": int(means_select.iloc[best_idx]["num"]), "tried": dict(df.count()["test"])}
            print("     ",filter_method, best_result)
            if best_result is None:
#                 print()
                results.append(None)
            else:
#                 print(round(best_result["test"], 2), "/", round(best_result["std"],2))
                results.append((round(best_result["test"], 2), round(best_result["std"],2)))
#         plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#         plt.legend()
        plt.show()
        all_results.append(results)
    return all_results


def to_latex(rank_methods, filter_methods, all_results):
    cmap = cm.get_cmap(name ='Blues') 
    max_delta = -100
    for col in all_results:
        for idx,row in enumerate(col):
            if row is not None:
                delta = row[0] - col[-1][0]
                if max_delta < delta:
                    max_delta = delta
                    
    cNorm  = colors.Normalize(vmin=0, vmax=max_delta*2)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
    print_names = {"gcn" : "GCN", 'simple' : "Common", 'adamic_ogb': "Adamic", 'simplecos':"Cosine-Common", 'sage':" SAGE", "None": "None (Baseline)"}
    latex_strs = []
    for filter_method in filter_methods:
        latex_strs.append("&" + print_names[filter_method])
    print(rank_methods)    
    for col in all_results:
        for idx,row in enumerate(col):
            if row is not None:
                color = colors.rgb2hex(scalarMap.to_rgba(row[0] - col[-1][0] ))
                if row[0]  <= col[-1][0]:
                    color = "ffffff"
                latex_strs[idx] = latex_strs[idx] + " & \cellcolor[HTML]{"+ color[-6:] +"}  $ "+ str(row[0])+"$ {\\tiny $\\pm "+ str(row[1])+ " $}"
            else:
                latex_strs[idx] = latex_strs[idx] + " & "
    for s in latex_strs:
        print(s +"\\\\")