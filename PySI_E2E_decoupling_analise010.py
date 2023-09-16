# ****************************
# Global Weekly PSI planning and simulation
# ****************************

# ****************************
# written by Yasushi Ohsugi with chatGPT
# as of 2023/09/16
# ****************************

# ****************************
# license : MIT license
# ****************************



# *********************************
# start of code
# *********************************
import pandas as pd
import csv

import math
import numpy as np

import datetime
import calendar


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# import plotly.graph_objs as go
import plotly.offline as offline
import plotly.io as pio

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import plotly.express as px


from copy import deepcopy

import itertools

import re


# *********************************
# for images directory
# *********************************
import os

# if not os.path.exists("temp_images"):
#    os.mkdir("temp_images")


# 幅優先探索 (Breadth-First Search)
from collections import deque


# **************************************
# 可視化トライアル
# **************************************

# node dictの在庫Iを可視化
def show_node_I4bullwhip_color(node_I4bullwhip):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # x, y, z軸のデータを作成
    x = np.arange(len(node_I4bullwhip["HAM_N"]))

    n = len(node_I4bullwhip.keys())
    y = np.arange(n)

    X, Y = np.meshgrid(x, y)

    z = list(node_I4bullwhip.keys())

    Z = np.zeros((n, len(x)))

    # node_I4bullwhipのデータをZに格納
    for i, node_name in enumerate(z):
        Z[i, :] = node_I4bullwhip[node_name]

    # 3次元の棒グラフを描画
    dx = dy = 1.2  # 0.8
    dz = Z
    colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
    for i in range(n):
        ax.bar3d(
            X[i],
            Y[i],
            np.zeros_like(dz[i]),
            dx,
            dy,
            dz[i],
            color=colors[i % len(colors)],
            alpha=0.8,
        )

    # 軸ラベルを設定
    ax.set_xlabel("Week")
    ax.set_ylabel("Node")
    ax.set_zlabel("Inventory")

    # y軸の目盛りをnode名に設定
    ax.set_yticks(y)
    ax.set_yticklabels(z)

    plt.show()


def show_psi_3D_graph_node(node):

    node_name = node.name

    # node_name = psi_list[0][0][0][:-7]
    # node_name = psiS2P[0][0][0][:-7]

    psi_list = node.psi4demand

    # 二次元マトリクスのサイズを定義する
    x_size = len(psi_list)
    y_size = len(psi_list[0])

    # x_size = len(psiS2P)
    # y_size = len(psiS2P[0])

    # x軸とy軸のグリッドを生成する
    x, y = np.meshgrid(range(x_size), range(y_size))

    # y軸の値に応じたカラーマップを作成
    color_map = plt.cm.get_cmap("cool")

    # z軸の値をリストから取得する
    z = []

    for i in range(x_size):
        row = []
        for j in range(y_size):

            row.append(len(psi_list[i][j]))
            # row.append(len(psiS2P[i][j]))

        z.append(row)

    ravel_z = np.ravel(z)

    norm = plt.Normalize(0, 3)
    # norm = plt.Normalize(0,dz.max())

    # 3Dグラフを作成する
    fig = plt.figure()

    ax = fig.add_subplot(111, projection="3d")

    z_like = np.zeros_like(z)

    # ********************
    # x/yの逆転
    # ********************
    original_matrix = z

    inverted_matrix = []

    for i in range(len(original_matrix[0])):
        inverted_row = []
        for row in original_matrix:
            inverted_row.append(row[i])
        inverted_matrix.append(inverted_row)

    z_inv = inverted_matrix

    # colors = plt.cm.terrain_r(norm(z_inv))
    # colors = plt.cm.terrain_r(norm(dz))

    # ********************
    # 4色での色分け
    # ********************

    # 色分け用のデータ
    color_data = [1, 2, 3, 4]

    # 色は固定
    # colorsのリストは、S/CO/I/Pに対応する
    # colors = ['cyan', 'blue', 'red', 'gold']
    # colors = ['cyan', 'blue', 'maroon', 'gold']
    colors = ["cyan", "blue", "brown", "gold"]

    y_list = np.ravel(y)

    c_map = []

    for index in y_list:

        c_map.append(colors[index])


    # ********************
    # bar3D
    # ********************

    ax.bar3d(
        np.ravel(x),
        np.ravel(y),
        np.ravel(np.zeros_like(z)),
        0.05,
        0.05,
        np.ravel(z_inv),
        color=c_map,
    )

    ax.set_title(node_name, fontsize="16")  # タイトル

    plt.show()


def visualise_psi_label(node_I_psi, node_name):

    # データの定義
    x, y, z = [], [], []

    for i in range(len(node_I_psi)):

        for j in range(len(node_I_psi[i])):

            # node_idx = node_name.index('JPN')

            node_label = node_name[i]  # 修正

            for k in range(len(node_I_psi[i][j])):
                x.append(j)
                y.append(node_label)
                z.append(k)

    text = []

    for i in range(len(node_I_psi)):

        for j in range(len(node_I_psi[i])):

            for k in range(len(node_I_psi[i][j])):

                text.append(node_I_psi[i][j][k])

    # y軸のラベルを設定
    y_axis = dict(tickvals=node_name, ticktext=node_name)

    # 3D散布図の作成
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                text=text,
                marker=dict(size=5, color=z, colorscale="Viridis", opacity=0.8),
            )
        ]
    )

    # レイアウトの設定
    fig.update_layout(
        title="Node Connections",
        scene=dict(xaxis_title="Week", yaxis_title="Location", zaxis_title="Lot ID"),
        width=800,
        height=800,
        margin=dict(l=65, r=50, b=65, t=90),
    )

    # グラフの表示
    # fig.show()
    return fig


# *****************************
# visualise I 3d bar
# *****************************
def visualise_inventory4demand_3d_bar(root_node, out_filename):

    nodes_list = []
    node_psI_list = []

    nodes_list, node_psI_list = extract_nodes_psI4demand(root_node)

    # *********************************
    # visualise with 3D bar graph
    # *********************************
    fig = visualise_psi_label(node_psI_list, nodes_list)

    offline.plot(fig, filename=out_filename)
    # offline.plot(fig, filename = out_filename)


def visualise_inventory4supply_3d_bar(root_node, out_filename):

    nodes_list = []
    node_psI_list = []
    plan_range = root_node.plan_range

    nodes_list, node_psI_list = extract_nodes_psI4supply(root_node, plan_range)

    # *********************************
    # visualise with 3D bar graph
    # *********************************
    fig = visualise_psi_label(node_psI_list, nodes_list)

    offline.plot(fig, filename=out_filename)


def visualise_I_bullwhip4supply(root_node, out_filename):

    plan_range = root_node.plan_range

    # *********************************
    # node_all_psiからIを抽出してnode_psI_list生成してvisualise
    # *********************************
    node_all_psi = {}

    node_all_psi = get_all_psi4supply(root_node, node_all_psi)

    # X
    week_len = 53 * plan_range + 1
    # week_len = len(node_yyyyww_lotid[0]) # node数が入る所を[0]で数える・・・

    # Y
    nodes_list = list(node_all_psi.keys())
    node_len = len(nodes_list)

    node_psI_list = [[] * i for i in range(node_len)]

    # **********************************
    # make bullwhip data    I_lot_step_week_node
    # **********************************

    # lot_stepの「値=長さ」の入れ物 x軸=week y軸=node
    #
    # week_len = len(node_yyyyww_lotid)ではなく 53 * plan_range でmaxに広げておく
    I_lot_step_week_node = [[None] * week_len for _ in range(node_len)]

    for node_name, psi_list in node_all_psi.items():

        node_index = nodes_list.index(node_name)

        supply_inventory_list = [[] * i for i in range(53 * plan_range)]
        # supply_inventory_list = [[]*i for i in range(len(psi_list))]

        for week in range(53 * plan_range):
            # for week in range(len(psi_list)):

            step_lots = psi_list[week][2]

            week_pos = week
            node_pos = nodes_list.index(node_name)

            I_lot_step_week_node[node_pos][week_pos] = len(step_lots)

            supply_inventory_list[week] = step_lots

        node_psI_list[node_index] = supply_inventory_list

    # ********************************
    # bullwhip visualise
    # ********************************
    I_visual_df = pd.DataFrame(I_lot_step_week_node, index=nodes_list)

    data = [go.Bar(x=I_visual_df.index, y=I_visual_df[0])]

    layout = go.Layout(
        title="Inventory Bullwip animation Global Supply Chain",
        xaxis={"title": "Location node"},
        yaxis={"title": "Lot-ID count", "showgrid": False},
        font={"size": 10},
        width=800,
        height=400,
        showlegend=False,
    )

    frames = []
    for week in I_visual_df.columns:
        frame_data = [go.Bar(x=I_visual_df.index, y=I_visual_df[week])]
        frame_layout = go.Layout(
            annotations=[
                go.layout.Annotation(
                    x=0.95,
                    y=1,
                    xref="paper",
                    yref="paper",
                    text=f"Week number: {week}",
                    showarrow=False,
                    font={"size": 14},
                )
            ]
        )
        frame = go.Frame(data=frame_data, layout=frame_layout)
        frames.append(frame)

    fig = go.Figure(data=data, layout=layout, frames=frames)

    offline.plot(fig, filename=out_filename)


def visualise_I_bullwhip4demand(root_node, out_filename):

    plan_range = root_node.plan_range

    # **********************************
    # make bullwhip data    I_lot_step_week_node
    # **********************************

    # *********************************
    # node_all_psiからIを抽出してnode_psI_list生成してvisualise
    # *********************************
    node_all_psi = {}

    node_all_psi = get_all_psi4demand(root_node, node_all_psi)

    # X
    week_len = 53 * plan_range + 1
    # week_len = len(node_yyyyww_lotid[0]) # node数が入る所を[0]で数える・・・


    # Y
    nodes_list = list(node_all_psi.keys())
    node_len = len(nodes_list)

    node_psI_list = [[] * i for i in range(node_len)]

    # **********************************
    # make bullwhip data    I_lot_step_week_node
    # **********************************

    # lot_stepの「値=長さ」の入れ物 x軸=week y軸=node

    I_lot_step_week_node = [[None] * week_len for _ in range(node_len)]

    for node_name, psi_list in node_all_psi.items():

        node_index = nodes_list.index(node_name)

        supply_inventory_list = [[] * i for i in range(53 * plan_range)]
        # supply_inventory_list = [[]*i for i in range(len(psi_list))]

        for week in range(53 * plan_range):

            step_lots = psi_list[week][2]

            week_pos = week
            node_pos = nodes_list.index(node_name)

            I_lot_step_week_node[node_pos][week_pos] = len(step_lots)

            supply_inventory_list[week] = step_lots

        node_psI_list[node_index] = supply_inventory_list

    # ********************************
    # bullwhip visualise
    # ********************************
    I_visual_df = pd.DataFrame(I_lot_step_week_node, index=nodes_list)

    data = [go.Bar(x=I_visual_df.index, y=I_visual_df[0])]

    layout = go.Layout(
        title="Inventory Bullwip animation Global Supply Chain",
        xaxis={"title": "Location node"},
        yaxis={"title": "Lot-ID count", "showgrid": False},
        font={"size": 10},
        width=800,
        height=400,
        showlegend=False,
    )

    frames = []
    for week in I_visual_df.columns:
        frame_data = [go.Bar(x=I_visual_df.index, y=I_visual_df[week])]
        frame_layout = go.Layout(
            annotations=[
                go.layout.Annotation(
                    x=0.95,
                    y=1,
                    xref="paper",
                    yref="paper",
                    text=f"Week number: {week}",
                    showarrow=False,
                    font={"size": 14},
                )
            ]
        )
        frame = go.Frame(data=frame_data, layout=frame_layout)
        frames.append(frame)

    fig = go.Figure(data=data, layout=layout, frames=frames)

    offline.plot(fig, filename=out_filename)


# ***********************************
# sub modules definition
# ***********************************


def extract_nodes_psI4demand(root_node):

    plan_range = root_node.plan_range

    # *********************************
    # node_all_psiからIを抽出してnode_psI_list生成してvisualise
    # *********************************
    node_all_psi = {}  # node:psi辞書に抽出

    node_all_psi = get_all_psi4demand(root_node, node_all_psi)
    # node_all_psi = get_all_psi4demand(root_node_outbound, node_all_psi)

    # get_all_psi4supply(root_node_outbound)


    # X
    week_len = 53 * plan_range + 1

    # Y
    nodes_list = list(node_all_psi.keys())

    node_len = len(nodes_list)

    node_psI_list = [[] * i for i in range(node_len)]

    for node_name, psi_list in node_all_psi.items():

        node_index = nodes_list.index(node_name)

        # supply_inventory_list = [[]*i for i in range( 53 * plan_range )]
        supply_inventory_list = [[] * i for i in range(len(psi_list))]

        for week in range(len(psi_list)):

            step_lots = psi_list[week][2]

            supply_inventory_list[week] = step_lots

        node_psI_list[node_index] = supply_inventory_list

    return nodes_list, node_psI_list


def extract_nodes_psI4demand_postorder(root_node):

    plan_range = root_node.plan_range

    # *********************************
    # node_all_psiからIを抽出してnode_psI_list生成してvisualise
    # *********************************
    node_all_psi = {}  # node:psi辞書に抽出

    node_all_psi = get_all_psi4demand_postorder(root_node, node_all_psi)
    # node_all_psi = get_all_psi4demand(root_node_outbound, node_all_psi)

    # get_all_psi4supply(root_node_outbound)

    week_len = 53 * plan_range + 1

    # Y
    nodes_list = list(node_all_psi.keys())

    node_len = len(nodes_list)

    node_psI_list = [[] * i for i in range(node_len)]

    for node_name, psi_list in node_all_psi.items():

        node_index = nodes_list.index(node_name)

        # supply_inventory_list = [[]*i for i in range( 53 * plan_range )]
        supply_inventory_list = [[] * i for i in range(len(psi_list))]

        for week in range(len(psi_list)):

            step_lots = psi_list[week][2]

            supply_inventory_list[week] = step_lots

        node_psI_list[node_index] = supply_inventory_list

    return nodes_list, node_psI_list


def extract_nodes_psI4supply(root_node, plan_range):
    # *********************************
    # node_all_psiからIを抽出してnode_psI_list生成してvisualise
    # *********************************
    node_all_psi = {}  # node:psi辞書に抽出

    node_all_psi = get_all_psi4supply(root_node, node_all_psi)

    # X
    week_len = 53 * plan_range + 1

    # Y
    nodes_list = list(node_all_psi.keys())

    node_len = len(nodes_list)

    node_psI_list = [[] * i for i in range(node_len)]

    for node_name, psi_list in node_all_psi.items():

        node_index = nodes_list.index(node_name)

        supply_inventory_list = [[] * i for i in range(len(psi_list))]

        for week in range(len(psi_list)):

            step_lots = psi_list[week][2]

            supply_inventory_list[week] = step_lots

        node_psI_list[node_index] = supply_inventory_list

    return nodes_list, node_psI_list


# 前処理として、年月の月間販売数の一日当たりの平均値を計算する
def calc_average_sales(monthly_sales, year):

    month_daily_average = [0] * 12

    for i, month_qty in enumerate(monthly_sales):

        month = i + 1

        days_in_month = calendar.monthrange(year, month)[1]

        month_daily_average[i] = monthly_sales[i] / days_in_month

    return month_daily_average


# *****************************************************
# ある年の月次販売数量を年月から年ISO週に変換する
# *****************************************************
def calc_weekly_sales(
    node,
    monthly_sales,
    year,
    year_month_daily_average,
    sales_by_iso_year,
    yyyyww_value,
    yyyyww_key,
):

    weekly_sales = [0] * 53

    for i, month_qty in enumerate(monthly_sales):

        # 開始月とリストの要素番号を整合
        month = i + 1

        # 月の日数を調べる
        days_in_month = calendar.monthrange(year, month)[1]

        # 月次販売の日平均
        avg_daily_sales = year_month_daily_average[year][i]  # i=month-1

        # 月の日毎の処理
        for day in range(1, days_in_month + 1):
            # その年の"年月日"を発生

            ## iso_week_noの確認 年月日でcheck その日がiso weekで第何週か
            # iso_week = datetime.date(year,month, day).isocalendar()[1]

            # ****************************
            # year month dayからiso_year, iso_weekに変換
            # ****************************
            dt = datetime.date(year, month, day)

            iso_year, iso_week, _ = dt.isocalendar()

            # 辞書に入れる場合
            sales_by_iso_year[iso_year][iso_week - 1] += avg_daily_sales

            # リストに入れる場合
            node_year_week_str = f"{node}{iso_year}{iso_week:02d}"

            if node_year_week_str not in yyyyww_key:

                yyyyww_key.append(node_year_week_str)

            pos = len(yyyyww_key) - 1

            yyyyww_value[pos] += avg_daily_sales

    return sales_by_iso_year[year]


# *******************************************************
# trans S from monthly to weekly
# *******************************************************
# 処理内容
# 入力ファイル: 拠点node別サプライチェーン需給tree
#               複数年別、1月-12月の需要数
#

# 処理        : iso_year+iso_weekをkeyにして、需要数を月間から週間に変換する

#               前処理で、各月の日数と月間販売数から、月毎の日平均値を求める
#               年月日からISO weekを判定し、
#               月間販売数の日平均値をISO weekの変数に加算、週間販売数を計算

#               ***** pointは「年月日からiso_year+iso_weekへの変換処理」 *****
#               dt = datetime.date(year, month, day)
#               iso_year, iso_week, _ = dt.isocalendar()

#               for nodeのループ下で、
#               YM_key_list.append(key)  ## keyをappendして
#               pos = len( YW_key_list ) ## YM_key_listの長さを位置にして
#               YW_value_list( pos ) += average_daily_value ## 値を+=加算

# 出力リスト  : node別 複数年のweekの需要 S_week


def trans_month2week(input_file, outputfile):

    # IN:      'S_month_data.csv'
    # PROCESS: nodeとyearを読み取る yearはstart-1年に"0"セットしてLT_shiftに備える
    # OUT:     'S_iso_week_data.csv'

    # *********************************
    # read monthly S
    # *********************************

    # csvファイルの読み込み
    df = pd.read_csv(input_file)  # IN:      'S_month_data.csv'

    #    # *********************************
    #    # mother plant capacity parameter
    #    # *********************************
    #
    #    demand_supply_ratio = 1.2  # demand_supply_ratio = ttl_supply / ttl_demand

    # *********************************
    # initial setting of total demand and supply
    # *********************************

    # total_demandは、各行のm1からm12までの列の合計値

    df_capa = pd.read_csv(input_file)

    df_capa["total_demand"] = df_capa.iloc[:, 3:].sum(axis=1)

    # yearでグループ化して、月次需要数の総和を計算
    df_capa_year = df_capa.groupby(["year"], as_index=False).sum()

    ## 結果を表示
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', None)
    # pd.set_option('display.max_colwidth', -1)

    # リストに変換
    month_data_list = df.values.tolist()

    # node_nameをユニークなキーとしたリストを作成する
    node_list = df["node_name"].unique().tolist()

    # *********************************
    # write csv file header [prod-A,node_name,year.w0,w1,w2,w3,,,w51,w52,w53]
    # *********************************

    file_name_out = outputfile  # OUT:     'S_iso_week_data.csv'

    with open(file_name_out, mode="w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "product_name",
                "node_name",
                "year",
                "w1",
                "w2",
                "w3",
                "w4",
                "w5",
                "w6",
                "w7",
                "w8",
                "w9",
                "w10",
                "w11",
                "w12",
                "w13",
                "w14",
                "w15",
                "w16",
                "w17",
                "w18",
                "w19",
                "w20",
                "w21",
                "w22",
                "w23",
                "w24",
                "w25",
                "w26",
                "w27",
                "w28",
                "w29",
                "w30",
                "w31",
                "w32",
                "w33",
                "w34",
                "w35",
                "w36",
                "w37",
                "w38",
                "w39",
                "w40",
                "w41",
                "w42",
                "w43",
                "w44",
                "w45",
                "w46",
                "w47",
                "w48",
                "w49",
                "w50",
                "w51",
                "w52",
                "w53",
            ]
        )

    # *********************************
    # plan initial setting
    # *********************************

    # node別に、中期計画の3ヵ年、5ヵ年をiso_year+iso_week連番で並べたもの
    # node_lined_iso_week = { node-A+year+week: [iso_year+iso_week1,2,3,,,,,],   }
    # 例えば、2024W00, 2024W01, 2024W02,,, ,,,2028W51,2028W52,2028W53という5年間分

    node_lined_iso_week = {}

    node_yyyyww_value = []
    node_yyyyww_key = []

    for node in node_list:

        df_node = df[df["node_name"] == node]

        # リストに変換
        node_data_list = df_node.values.tolist()

        #
        # getting start_year and end_year
        #
        start_year = node_data_min = df_node["year"].min()
        end_year = node_data_max = df_node["year"].max()

        # S_month辞書の初期セット
        monthly_sales_data = {}

        # *********************************
        # plan initial setting
        # *********************************

        plan_year_st = start_year  # 2024  # plan開始年

        plan_range = end_year - start_year + 1  # 5     # 5ヵ年計画分のS計画

        plan_year_end = plan_year_st + plan_range

        #
        # an image of data "df_node"
        #
        # product_name	node_name	year	m1	m2	m3	m4	m5	m6	m7	m8	m9	m10	m11	m12
        # prod-A	CAN	2024	0	0	0	0	0	0	0	0	0	0	0	0
        # prod-A	CAN	2025	0	0	0	0	0	0	0	0	0	0	0	0
        # prod-A	CAN	2026	0	0	0	0	0	0	0	0	0	0	0	0
        # prod-A	CAN	2027	0	0	0	0	0	0	0	0	0	0	0	0
        # prod-A	CAN	2028	0	0	0	0	0	0	0	0	0	0	0	0
        # prod-A	CAN_D	2024	122	146	183	158	171	195	219	243	231	207	195	219
        # prod-A	CAN_D	2025	122	146	183	158	171	195	219	243	231	207	195	219

        # *********************************
        # by node    node_yyyyww = [ node-a, yyyy01, yyyy02,,,, ]
        # *********************************

        yyyyww_value = [0] * 53 * plan_range  # 5ヵ年plan_range=5

        yyyyww_key = []

        for data in node_data_list:

            # node別　3年～5年　月次需要予測値

            # 辞書形式{year: S_week_list, }でデータ定義する
            sales_by_iso_year = {}

            # 前後年付きの辞書 53週を初期セット
            # **********************************
            # 空リストの初期設定
            # start and end setting from S_month data # 月次Sのデータからmin&max
            # **********************************

            # 前年の52週が発生する可能性あり # 計画の前後の-1年 +1年を見る
            work_year = plan_year_st - 1

            for i in range(plan_range + 2):  # 計画の前後の-1年 +1年を見る

                year_sales = [0] * 53  # 53週分の要素を初期セット

                # 年の辞書に週次Sをセット
                sales_by_iso_year[work_year] = year_sales

                work_year += 1

            # *****************************************
            # initial setting end
            # *****************************************

            # *****************************************
            # start process
            # *****************************************

            # ********************************
            # generate weekly S from monthly S
            # ********************************

            # S_monthのcsv fileを読んでS_month_listを生成する
            # pandasでcsvからリストにして、node_nameをキーに順にM2W変換

            # ****************** year ****** Smonth_list ******
            monthly_sales_data[data[2]] = data[3:]

            # data[0] = prod-A
            # data[1] = node_name
            # data[2] = year

        # **************************************
        # 年月毎の販売数量の日平均を計算する
        # **************************************
        year_month_daily_average = {}

        for y in range(plan_year_st, plan_year_end):

            year_month_daily_average[y] = calc_average_sales(monthly_sales_data[y], y)

        # 販売数量を年月から年ISO週に変換する
        for y in range(plan_year_st, plan_year_end):

            sales_by_iso_year[y] = calc_weekly_sales(
                node,
                monthly_sales_data[y],
                y,
                year_month_daily_average,
                sales_by_iso_year,
                yyyyww_value,
                yyyyww_key,
            )

        work_yyyyww_value = [node] + yyyyww_value
        work_yyyyww_key = [node] + yyyyww_key

        node_yyyyww_value.append(work_yyyyww_value)
        node_yyyyww_key.append(work_yyyyww_key)

        # 複数年のiso週毎の販売数を出力する
        for y in range(plan_year_st, plan_year_end):

            rowX = ["product-X"] + [node] + [y] + sales_by_iso_year[y]

            with open(file_name_out, mode="a", newline="") as f:

                writer = csv.writer(f)

                writer.writerow(rowX)

    # **********************
    # リスト形式のS出力
    # **********************

    return node_yyyyww_value, node_yyyyww_key, plan_range, df_capa_year


# *********************
# END of week data generation
# node_yyyyww_value と node_yyyyww_keyに複数年の週次データがある
# *********************


# *******************************************************
# lot by lot PSI
# *******************************************************
def makeS(S_week, lot_size):  # Sの値をlot単位に変換してリスト化

    return [math.ceil(num / lot_size) for num in S_week]


# @230908 mark このlotid生成とセットするpsi_listを、直接tree上のpsiのS[0]に置く
# @230908 mark または、S_stackedというリストで返す


def make_lotid_stack(S_stack, node_name, Slot, node_yyyyww_list):

    for w, (lots_count, node_yyyyww) in enumerate(zip(Slot, node_yyyyww_list)):

        stack_list = []

        for i in range(lots_count):

            lot_id = str(node_yyyyww) + str(i)

            stack_list.append(lot_id)

        ## week 0="S"
        # psi_list[w][0] = stack_list

        S_stack[w] = stack_list

    return S_stack


def make_lotid_list_setS(psi_list, node_name, Slot, yyyyww_list):

    for w, (lots_count, yyyyww) in enumerate(zip(Slot, yyyyww_list)):

        stack_list = []

        for i in range(lots_count):

            lot_id = str(yyyyww) + str(i)

            stack_list.append(lot_id)

        psi_list[w][0] = stack_list

    return psi_list


# ************************************
# checking constraint to inactive week , that is "Long Vacation"
# ************************************
def check_lv_week_bw(const_lst, check_week):

    num = check_week

    if const_lst == []:

        pass

    else:

        while num in const_lst:

            num -= 1

    return num


def check_lv_week_fw(const_lst, check_week):

    num = check_week

    if const_lst == []:

        pass

    else:

        while num in const_lst:

            num += 1

    return num


def calcPS2I4demand(psiS2P):

    plan_len = len(psiS2P)

    for w in range(1, plan_len):  # starting_I = 0 = w-1 / ending_I = 53
        # for w in range(1,54): # starting_I = 0 = w-1 / ending_I = 53

        s = psiS2P[w][0]
        co = psiS2P[w][1]

        i0 = psiS2P[w - 1][2]
        i1 = psiS2P[w][2]

        p = psiS2P[w][3]

        # *********************
        # # I(n-1)+P(n)-S(n)
        # *********************

        work = i0 + p

        # memo ここで、期末の在庫、S出荷=売上を操作している
        # S出荷=売上を明示的にlogにして、price*qty=rev売上として記録し表示処理
        # 出荷されたS=売上、在庫I、未出荷COの集合を正しく表現する

        # **************************
        # モノがお金に代わる瞬間
        # **************************

        diff_list = [x for x in work if x not in s]  # I(n-1)+P(n)-S(n)

        psiS2P[w][2] = i1 = diff_list

    return psiS2P


def shiftS2P_LV(psiS, safety_stock_week, lv_week):  # LV:long vacations

    ss = safety_stock_week

    plan_len = len(psiS) - 1  # -1 for week list position

    for w in range(plan_len, ss, -1):  # backward planningで需要を降順でシフト

        # my_list = [1, 2, 3, 4, 5]
        # for i in range(2, len(my_list)):
        #    my_list[i] = my_list[i-1] + my_list[i-2]

        # 0:S
        # 1:CO
        # 2:I
        # 3:P

        eta_plan = w - ss  # ss:safty stock

        eta_shift = check_lv_week_bw(lv_week, eta_plan)  # ETA:Eatimate Time Arrival

        # リスト追加 extend
        # 安全在庫とカレンダ制約を考慮した着荷予定週Pに、w週Sからoffsetする

        psiS[eta_shift][3].extend(psiS[w][0])  # P made by shifting S with

    return psiS


def shiftS2P_LV_replace(psiS, safety_stock_week, lv_week):  # LV:long vacations

    ss = safety_stock_week

    plan_len = len(psiS) - 1  # -1 for week list position

    for w in range(plan_len):  # foreward planningでsupplyのp [w][3]を初期化

        # psiS[w][0] = [] # S active

        psiS[w][1] = []  # CO
        psiS[w][2] = []  # I
        psiS[w][3] = []  # P

    for w in range(plan_len, ss, -1):  # backward planningでsupplyを降順でシフト

        # my_list = [1, 2, 3, 4, 5]
        # for i in range(2, len(my_list)):
        #    my_list[i] = my_list[i-1] + my_list[i-2]

        # 0:S
        # 1:CO
        # 2:I
        # 3:P

        eta_plan = w - ss  # ss:safty stock

        eta_shift = check_lv_week_bw(lv_week, eta_plan)  # ETA:Eatimate Time Arrival

        # リスト追加 extend
        # 安全在庫とカレンダ制約を考慮した着荷予定週Pに、w週Sからoffsetする
        psiS[eta_shift][3].extend(psiS[w][0])  # P made by shifting S with

    return psiS



def shiftP2S_LV(psiP, safety_stock_week, lv_week):  # LV:long vacations

    ss = safety_stock_week

    plan_len = len(psiP) - 1  # -1 for week list position

    for w in range(plan_len - 1):  # forward planningで確定Pを確定Sにシフト

        # my_list = [1, 2, 3, 4, 5]
        # for i in range(2, len(my_list)):
        #    my_list[i] = my_list[i-1] + my_list[i-2]

        # 0:S
        # 1:CO
        # 2:I
        # 3:P

        etd_plan = w + ss  # ss:safty stock

        etd_shift = check_lv_week_fw(lv_week, etd_plan)  # ETD:Eatimate TimeDep
        # リスト追加 extend
        # 安全在庫とカレンダ制約を考慮した着荷予定週Pに、w週Sからoffsetする

        psiP[etd_shift][0] = psiP[w][3]  # S made by shifting P with

    return psiP


def make_S_lots(node_yyyyww_value, node_yyyyww_key, nodes):

    S_lots_dict = {}

    for i, node_val in enumerate(node_yyyyww_value):  # by nodeでrepeat処理

        node_name = node_val[0]
        S_week = node_val[1:]

        node = nodes[node_name]  # node_nameからnodeインスタンスを取得

        # node.lot_sizeを使う
        lot_size = node.lot_size  # Node()からセット

        # makeSでSlotを生成
        # ロット数に変換し、週リストで返す # lotidではない
        # return [math.ceil(num / lot_size) for num in S_week]

        # Slot = makeS(S_week, lot_size)
        Slot = [math.ceil(num / lot_size) for num in S_week]

        ## nodeに対応するpsi_list[w][0,1,2,3]を生成する
        # psi_list = [[[] for j in range(4)] for w in range( len(S_week) )]

        S_stack = [[] for w in range(len(S_week))]

        node_key = node_yyyyww_key[i]  # node_name + yyyyww

        ####node_name = node_key[0] # node_valと同じ

        yyyyww_list = node_key[1:]

        # lotidをリスト化 #  Slotの要素「ロット数」からlotidを付番してリスト化
        S_lots_dict[node.name] = make_lotid_stack(S_stack, node_name, Slot, yyyyww_list)

    return S_lots_dict


def make_node_psi_dict(node_yyyyww_value, node_yyyyww_key, nodes):

    node_psi_dict = {}  # node_psi辞書

    for i, node_val in enumerate(node_yyyyww_value):  # by nodeでrepeat処理

        node_name = node_val[0]
        S_week = node_val[1:]

        node = nodes[node_name]  # node_nameからnodeインスタンスを取得

        # node.lot_sizeを使う
        lot_size = node.lot_size  # Node()からセット

        # makeSでSlotを生成
        # ロット数に変換し、週リストで返す # lotidではない
        # return [math.ceil(num / lot_size) for num in S_week]

        Slot = makeS(S_week, lot_size)

        # nodeに対応するpsi_list[w][0,1,2,3]を生成する
        psi_list = [[[] for j in range(4)] for w in range(len(S_week))]

        node_key = node_yyyyww_key[i]  # node_name + yyyyww

        yyyyww_list = node_key[1:]

        # lotidをリスト化 #  Slotの要素「ロット数」からlotidを付番してリスト化
        psiS = make_lotid_list_setS(psi_list, node_name, Slot, yyyyww_list)

        node_psi_dict[node_name] = psiS  # 初期セットSを渡す。本来はleaf_nodeのみ

    return node_psi_dict


# ***************************************
# mother plant/self.nodeの確定Sから子nodeを分離
# ***************************************
def extract_node_conf(req_plan_node, S_confirmed_plan):

    node_list = list(itertools.chain.from_iterable(req_plan_node))

    extracted_list = []
    extracted_list.extend(S_confirmed_plan)

    # フラットなリストに展開する
    flattened_list = [item for sublist in extracted_list for item in sublist]

    # node_listとextracted_listを比較して要素の追加と削除を行う
    extracted_list = [
        [item for item in sublist if item in node_list] for sublist in extracted_list
    ]

    return extracted_list


def separated_node_plan(node_req_plans, S_confirmed_plan):

    shipping_plans = []

    for req_plan in node_req_plans:

        shipping_plan = extract_node_conf(req_plan, S_confirmed_plan)

        shipping_plans.append(shipping_plan)

    return shipping_plans


# **********************************
# create tree
# **********************************
class Node:  # with "parent"
    def __init__(self, name):
        self.name = name
        self.children = []
        self.parent = None

        # application attribute # nodeをインスタンスした後、初期値セット
        self.psi4demand = None
        self.psi4supply = None

        self.psi4couple = None

        self.psi4accume = None

        self.plan_range = 1
        self.plan_year_st = 2020

        self.safety_stock_week = 0
        # self.safety_stock_week = 2

        # self.lv_week = []

        self.lot_size = 1  # defalt set

        # leadtimeとsafety_stock_weekは、ここでは同じ
        self.leadtime = 1  # defalt set  # 前提:SS=0

        self.long_vacation_weeks = []

        # evaluation
        self.decoupling_total_I = []  # total Inventory all over the plan

    def add_child(self, child):
        self.children.append(child)

    def set_parent(self):
        # def set_parent(self, node):

        # treeを辿りながら親ノードを探索
        if self.children == []:

            pass

        else:

            for child in self.children:

                child.parent = self
                # child.parent = node

    def set_attributes(self, row):
        self.lot_size = int(row[3])
        self.leadtime = int(row[4])  # 前提:SS=0
        self.long_vacation_weeks = eval(row[5])

    def set_psi_list(self, psi_list):

        self.psi4demand = psi_list

    # supply_plan
    def set_psi_list4supply(self, psi_list):

        self.psi4supply = psi_list


    def get_set_childrenP2S2psi(self, plan_range):

        for child in self.children:

            for w in range(53 * plan_range):

                self.psi4demand[w][0].extend(child.psi4demand[w][3])



    def confirmedS2childrenP_by_lot(self, plan_range):

        # マザープラントの確定したSを元に、
        # demand_plan上のlot_idの状態にかかわらず、
        # supply_planにおいては、
        # 確定出荷confirmed_Sを元に、以下のpush planを実行する

        # by lotidで一つずつ処理する。

        # 親のconfSのlotidは、どの子nodeから来たのか、出荷先を特定する。
        #  "demand_planのpsi_listのS" の中をサーチしてisin.listで特定する。
        # search_node()では子node psiの中にlotidがあるかisinでcheck

        # LT_shiftして、子nodeのPにplaceする。
        # 親S_lotid => ETA=LT_shift() =>子P[ETA][3]

        # 着荷PをSS_shiftして出荷Sをセット
        # 子P=>SS_shift(P)=>子S

        # Iの生成
        # all_PS2I

        # 親の確定出荷confirmedSをを子供の確定Pとして配分
        # 最後に、conf_PをLT_shiftしてconf_Sにもセットする
        # @230717このLT_shiftの中では、cpnf_Sはmoveする/extendしない

        #
        # def feedback_confirmedS2childrenP(self, plan_range):
        #

        self_confirmed_plan = [[] for _ in range(53 * plan_range)]

        # ************************************
        # setting mother_confirmed_plan
        # ************************************
        for w in range(53 * plan_range):

            ## 親node自身のsupply_planのpsi_list[w][0]がconfirmed_S
            # self_confirmed_plan[w].extend(self.psi4supply[w][0])

            confirmed_S_lots = self.psi4supply[w][0]

            for lot in confirmed_S_lots:

                if lot == []:

                    pass

                else:

                    # @230722 lot by lot operation

                    ## in parent
                    # get_lot()

                    # in children
                    # demand側で出荷先の子nodeを確認し
                    node_to = check_lot_in_demand_plan(lot)

                    # supply側で出荷先の子nodeに置く
                    # LT_shiftしてset_P
                    # SS_shiftしてset_S
                    place_lot_in_supply_plan(node_to)

        # end of
        #    def confirmedS2childrenP_by_lot(self, plan_range):

        node_req_plans = []
        node_confirmed_plans = []


        # ************************************
        # setting node_req_plans 各nodeからの要求S(=P)
        # ************************************
        # 子nodeのdemand_planのpsi_list[w][3]のPがS_requestに相当する

        # すべての子nodesから、S_reqをappendしてnode_req_plansを作る
        for child in self.children:

            child_S_req = [[] for _ in range(53 * plan_range)]

            for w in range(53 * plan_range):

                child_S_req[w].extend(child.psi4demand[w][3])  # setting P2S

            node_req_plans.append(child_S_req)

        # node_req_plans      子nodeのP=S要求計画planのリストplans
        # self_confirmed_plan 自nodeの供給計画の確定S

        # 出荷先ごとの出荷計画を求める
        # node_req_plans = [req_plan_node_1, req_plan_node_2, req_plan_node_3]

        # ***************************
        # node 分離
        # ***************************
        node_confirmed_plans = []

        node_confirmed_plans = separated_node_plan(node_req_plans, self_confirmed_plan)

        for i, child in enumerate(self.children):

            for w in range(53 * plan_range):

                # 子nodeのsupply_planのPにmother_plantの確定Sをセット

                child.psi4supply[w][3] = []  # clearing list

                # i番目の子nodeの確定Sをsupply_planのPにextendでlot_idをcopy

                child.psi4supply[w][3].extend(node_confirmed_plans[i][w])

            # ココまででsupply planの子nodeにPがセットされたことになる。

        # *******************************************
        # supply_plan上で、PfixをSfixにPISでLT offsetする
        # *******************************************

        # **************************
        # Safety Stock as LT shift
        # **************************
        safety_stock_week = self.leadtime

        # 案-1:長い搬送工程はPSI処理の対象とし、self.LT=搬送LT(+SS=2)とする
        #      生産工程は、self.LT=加工LT  SS=0
        #      保管工程は、self.LT=入出庫LT SS=0

        # 案-2:
        # process_week = self.process_leadtime
        # safety_stock_week = self.SS_leadtime

        # demand plan : using "process_LT" + "safety_stock_LT" with backward planning
        # supply plan : using "process_LT"                     with foreward planning

        # **************************
        # long vacation weeks
        # **************************
        lv_week = self.long_vacation_weeks

        # P to S の計算処理
        self.psi4supply = shiftP2S_LV(self.psi4supply, safety_stock_week, lv_week)

        ## S to P の計算処理
        # self.psi4demand = shiftS2P_LV(self.psi4demand, safety_stock_week, lv_week)

    def set_S2psi(self, pSi):
        # def set_S2psi(self, S_lots_list):
        # def set_Slots2psi4demand(self, S_lots_list):

        # S_lots_listが辞書で、node.psiにセットする

        for w in range(len(pSi)):
            self.psi4demand[w][0].extend(pSi[w])



    def feedback_confirmedS2childrenP(self, plan_range):

        # マザープラントの確定したSを元に、
        # demand_plan上のlot_idの状態にかかわらず、
        # supply_planにおいては、
        # 確定出荷confirmed_Sを元に、以下のpush planを実行する

        # by lotidで一つずつ処理する。

        # 親のconfSのlotidは、どの子nodeから来たのか?
        #  "demand_planのpsi_listのS" の中をサーチしてisin.listで特定する。
        # search_node()では子node psiの中にlotidがあるかisinでcheck

        # LT_shiftして、子nodeのPにplaceする。
        # 親S_lotid => ETA=LT_shift() =>子P[ETA][3]

        # 着荷PをSS_shiftして出荷Sをセット
        # 子P=>SS_shift(P)=>子S

        # Iの生成
        # all_PS2I

        # 親の確定出荷confirmedSをを子供の確定Pとして配分
        # 最後に、conf_PをLT_shiftしてconf_Sにもセットする
        # @230717このLT_shiftの中では、cpnf_Sはmoveする/extendしない

        #
        # def feedback_confirmedS2childrenP(self, plan_range):
        #
        node_req_plans = []
        node_confirmed_plans = []

        self_confirmed_plan = [[] for _ in range(53 * plan_range)]

        # ************************************
        # setting mother_confirmed_plan
        # ************************************
        for w in range(53 * plan_range):

            # 親node自身のsupply_planのpsi_list[w][0]がconfirmed_S
            self_confirmed_plan[w].extend(self.psi4supply[w][0])

        # ************************************
        # setting node_req_plans 各nodeからの要求S(=P)
        # ************************************
        # 子nodeのdemand_planのpsi_list[w][3]のPがS_requestに相当する

        # すべての子nodesから、S_reqをappendしてnode_req_plansを作る
        for child in self.children:

            child_S_req = [[] for _ in range(53 * plan_range)]

            for w in range(53 * plan_range):

                child_S_req[w].extend(child.psi4demand[w][3])  # setting P2S

            node_req_plans.append(child_S_req)

        # node_req_plans      子nodeのP=S要求計画planのリストplans
        # self_confirmed_plan 自nodeの供給計画の確定S

        # 出荷先ごとの出荷計画を求める
        # node_req_plans = [req_plan_node_1, req_plan_node_2, req_plan_node_3]

        # ***************************
        # node 分離
        # ***************************
        node_confirmed_plans = []

        node_confirmed_plans = separated_node_plan(node_req_plans, self_confirmed_plan)

        for i, child in enumerate(self.children):

            for w in range(53 * plan_range):

                # 子nodeのsupply_planのPにmother_plantの確定Sをセット

                child.psi4supply[w][3] = []  # clearing list

                # i番目の子nodeの確定Sをsupply_planのPにextendでlot_idをcopy

                child.psi4supply[w][3].extend(node_confirmed_plans[i][w])

            # ココまででsupply planの子nodeにPがセットされたことになる。

        # *******************************************
        # supply_plan上で、PfixをSfixにPISでLT offsetする
        # *******************************************

        # **************************
        # Safety Stock as LT shift
        # **************************
        safety_stock_week = self.leadtime

        # **************************
        # long vacation weeks
        # **************************
        lv_week = self.long_vacation_weeks

        # P to S の計算処理
        self.psi4supply = shiftP2S_LV(self.psi4supply, safety_stock_week, lv_week)

        ## S to P の計算処理
        # self.psi4demand = shiftS2P_LV(self.psi4demand, safety_stock_week, lv_week)

    def calcPS2I4demand(self):

        # psiS2P = self.psi4demand # copyせずに、直接さわる

        plan_len = 53 * self.plan_range
        # plan_len = len(self.psi4demand)

        for w in range(1, plan_len):  # starting_I = 0 = w-1 / ending_I =plan_len
            # for w in range(1,54): # starting_I = 0 = w-1 / ending_I = 53

            s = self.psi4demand[w][0]
            co = self.psi4demand[w][1]

            i0 = self.psi4demand[w - 1][2]
            i1 = self.psi4demand[w][2]

            p = self.psi4demand[w][3]

            # *********************
            # # I(n-1)+P(n)-S(n)
            # *********************

            work = i0 + p  # 前週在庫と当週着荷分 availables

            # ここで、期末の在庫、S出荷=売上を操作している
            # S出荷=売上を明示的にlogにして、売上として記録し、表示する処理
            # 出荷されたS=売上、在庫I、未出荷COの集合を正しく表現する

            # モノがお金に代わる瞬間

            diff_list = [x for x in work if x not in s]  # I(n-1)+P(n)-S(n)

            self.psi4demand[w][2] = i1 = diff_list

    def calcPS2I4supply(self):

        # psiS2P = self.psi4demand # copyせずに、直接さわる

        plan_len = 53 * self.plan_range
        # plan_len = len(self.psi4supply)

        for w in range(1, plan_len):  # starting_I = 0 = w-1 / ending_I =plan_len
            # for w in range(1,54): # starting_I = 0 = w-1 / ending_I = 53

            s = self.psi4supply[w][0]
            co = self.psi4supply[w][1]

            i0 = self.psi4supply[w - 1][2]
            i1 = self.psi4supply[w][2]

            p = self.psi4supply[w][3]

            # *********************
            # # I(n-1)+P(n)-S(n)
            # *********************

            work = i0 + p  # 前週在庫と当週着荷分 availables

            # memo ここで、期末の在庫、S出荷=売上を操作している
            # S出荷=売上を明示的にlogにして、売上として記録し、表示する処理
            # 出荷されたS=売上、在庫I、未出荷COの集合を正しく表現する

            # モノがお金に代わる瞬間

            diff_list = [x for x in work if x not in s]  # I(n-1)+P(n)-S(n)

            self.psi4supply[w][2] = i1 = diff_list

    def calcPS2I_decouple4supply(self):

        # psiS2P = self.psi4demand # copyせずに、直接さわる

        plan_len = 53 * self.plan_range
        # plan_len = len(self.psi4supply)

        # demand planのSを出荷指示情報=PULL SIGNALとして、supply planSにセット

        for w in range(0, plan_len):
            # for w in range(1,plan_len):

            # pointer参照していないか? 明示的にデータを渡すには?

            self.psi4supply[w][0] = self.psi4demand[w][
                0
            ].copy()  # copy data using copy() method
            # self.psi4supply[w][0]    = self.psi4demand[w][0] # PULL replaced

            #checking pull data
            #show_psi_graph(root_node_outbound,"supply", "HAM", 0, 300 )
            #show_psi_graph(root_node_outbound,"supply", node_show, 0, 300 )

        for w in range(1, plan_len):  # starting_I = 0 = w-1 / ending_I =plan_len

            # demand planSをsupplySにコピー済み
            s = self.psi4supply[w][0]  # PUSH supply S

            co = self.psi4supply[w][1]

            i0 = self.psi4supply[w - 1][2]
            i1 = self.psi4supply[w][2]

            p = self.psi4supply[w][3]

            # *********************
            # # I(n-1)+P(n)-S(n)
            # *********************

            work = i0 + p  # 前週在庫と当週着荷分 availables

            # memo ここで、期末の在庫、S出荷=売上を操作している
            # S出荷=売上を明示的にlogにして、売上として記録し、表示する処理
            # 出荷されたS=売上、在庫I、未出荷COの集合を正しく表現する

            # モノがお金に代わる瞬間

            diff_list = [x for x in work if x not in s]  # I(n-1)+P(n)-S(n)

            self.psi4supply[w][2] = i1 = diff_list



    def calcS2P(self):

        # **************************
        # Safety Stock as LT shift
        # **************************
        # leadtimeとsafety_stock_weekは、ここでは同じ
        safety_stock_week = self.leadtime

        # **************************
        # long vacation weeks
        # **************************
        lv_week = self.long_vacation_weeks

        # S to P の計算処理
        self.psi4demand = shiftS2P_LV(self.psi4demand, safety_stock_week, lv_week)

        pass

    def calcS2P_4supply(self):

        # **************************
        # Safety Stock as LT shift
        # **************************
        # leadtimeとsafety_stock_weekは、ここでは同じ
        safety_stock_week = self.leadtime

        # **************************
        # long vacation weeks
        # **************************
        lv_week = self.long_vacation_weeks

        # S to P の計算処理
        self.psi4supply = shiftS2P_LV_replace(
            self.psi4supply, safety_stock_week, lv_week
        )

        pass


# ****************************
# supply chain tree creation
# ****************************
def create_tree(csv_file):

    root_node_name = ""  # init setting

    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        next(reader)  # ヘッダー行をスキップ

        # nodeインスタンスの辞書を作り、親子の定義に使う
        nodes = {row[2]: Node(row[2]) for row in reader}

        f.seek(0)  # ファイルを先頭に戻す

        next(reader)  # ヘッダー行をスキップ

        # next(reader)  # root行をスキップしないでloopに入る

        # readerの一行目root rawのroot_node_nameを取得する

        for row in reader:

            if row[0] == "root":

                root_node_name = row[1]

            else:

                parent = nodes[row[0]]

                child = nodes[row[1]]

                parent.add_child(child)

                child.set_attributes(row)  # 子ノードにアプリケーション属性セット

    return nodes, root_node_name  # すべてのインスタンス・ポインタを返して使う
    # return nodes['JPN']   # "JPN"のインスタンス・ポインタ


def set_psi_lists(node, node_psi_dict):
    # キーが存在する場合は対応する値valueが返り、存在しない場合はNoneが返る。
    if node.children == []:  # 子nodeがないleaf nodeの場合

        node.set_psi_list(node_psi_dict.get(node.name))

    else:

        node.get_set_childrenP2S2psi(node.plan_range)

    for child in node.children:

        set_psi_lists(child, node_psi_dict)


def set_Slots2psi4OtDm(node, S_lots_list):

    for child in node.children:

        set_Slots2psi4OtDm(child, S_lots_list)

    # キーが存在する場合は対応する値valueが返り、存在しない場合はNoneが返る。
    if node.children == []:  # 子nodeがないleaf nodeの場合

        # S_lots_listが辞書で、node.psiにセットする
        node.set_S2psi(S_lots_list)

        # shifting S2P
        node.calcS2P()  # backward plan with postordering


    else:

        # gathering S and Setting S
        node.get_set_childrenP2S2psi(node.plan_range)
        # node.get_set_childrenS2psi(plan_range)

        # shifting S2P
        node.calcS2P()  # backward plan with postordering


# _dictで渡す
def set_Slots2psi4demand(node, S_lots_dict):
    # def set_psi_lists_postorder(node, node_psi_dict):

    for child in node.children:

        set_Slots2psi4demand(child, S_lots_dict)

    # キーが存在する場合は対応する値valueが返り、存在しない場合はNoneが返る。
    if node.children == []:  # 子nodeがないleaf nodeの場合

        # 辞書のgetメソッドでキーから値を取得。キーが存在しない場合はNone
        pSi = S_lots_dict.get(node.name)

        node.set_S2psi(pSi)

        # shifting S2P
        node.calcS2P()  # backward plan with postordering

    else:

        # gathering S and Setting S
        node.get_set_childrenP2S2psi(node.plan_range)

        # shifting S2P
        node.calcS2P()  # backward plan with postordering


def set_psi_lists_postorder(node, node_psi_dict):

    for child in node.children:

        set_psi_lists_postorder(child, node_psi_dict)

    # キーが存在する場合は対応する値valueが返り、存在しない場合はNoneが返る。
    if node.children == []:  # 子nodeがないleaf nodeの場合

        # 辞書のgetメソッドでキーから値を取得。キーが存在しない場合はNone
        node.set_psi_list(node_psi_dict.get(node.name))

        # shifting S2P
        node.calcS2P()  # backward plan with postordering


    else:

        # gathering S and Setting S
        node.get_set_childrenP2S2psi(node.plan_range)
        # node.get_set_childrenS2psi(plan_range)

        # shifting S2P
        node.calcS2P()  # backward plan with postordering


def make_psi4supply(node, node_psi_dict):

    plan_range = node.plan_range

    node_psi_dict[node.name] = [[[] for j in range(4)] for w in range(53 * plan_range)]

    for child in node.children:

        make_psi4supply(child, node_psi_dict)

    return node_psi_dict


def set_psi_lists4supply(node, node_psi_dict):

    node.set_psi_list4supply(node_psi_dict.get(node.name))

    for child in node.children:

        set_psi_lists4supply(child, node_psi_dict)


def find_path_to_leaf_with_parent(node, leaf_node, current_path=[]):

    current_path.append(leaf_node.name)

    if node.name == leaf_node.name:

        return current_path

    else:

        parent = leaf_node.parent

        path = find_path_to_leaf_with_parent(node, parent, current_path.copy())

    return path


#        if path:
#
#            return path


def find_path_to_leaf(node, leaf_node, current_path=[]):

    current_path.append(node.name)

    if node.name == leaf_node.name:

        return current_path

    for child in node.children:

        path = find_path_to_leaf(child, leaf_node, current_path.copy())

        if path:

            return path



def flatten_list(data_list):
    for item in data_list:
        if isinstance(item, list):
            yield from flatten_list(item)
        else:
            yield item


def children_nested_list(data_list):

    flat_list = set(flatten_list(data_list))

    return flat_list



def extract_node_name(stringA):
    # 右側の数字部分を除外してnode名を取得

    index = len(stringA) - 1

    while index >= 0 and stringA[index].isdigit():

        index -= 1

    node_name = stringA[: index + 1]

    return node_name



def place_P_in_supply(w, child, lot):  # lot LT_shift on P

    # *******************************************
    # supply_plan上で、PfixをSfixにPISでLT offsetする
    # *******************************************

    # **************************
    # Safety Stock as LT shift
    # **************************

    # leadtimeとsafety_stock_weekは、ここでは同じ
    # safety_stock_week = child.leadtime
    LT_SS_week = child.leadtime

    # **************************
    # long vacation weeks
    # **************************
    lv_week = child.long_vacation_weeks

    ## P to S の計算処理
    # self.psi4supply = shiftP2S_LV(self.psi4supply, safety_stock_week, lv_week)

    ### S to P の計算処理
    ##self.psi4demand = shiftS2P_LV(self.psi4demand, safety_stock_week, lv_week)

    # my_list = [1, 2, 3, 4, 5]
    # for i in range(2, len(my_list)):
    #    my_list[i] = my_list[i-1] + my_list[i-2]

    # 0:S
    # 1:CO
    # 2:I
    # 3:P

    # @230723モデルの基本的な定義について
    # このモデルの前提では、輸送工程をPSI計算しているので、
    # ETD=ETAとなっている。。。不自然???

    # LT:leadtime SS:safty stockは1つ
    # foreward planで、「親confirmed_S出荷=子confirmed_P着荷」と表現
    eta_plan = w + LT_SS_week  # ETA=ETDなので、+LTすると次のETAとなる

    # etd_plan = w + ss # ss:safty stock
    # eta_plan = w - ss # ss:safty stock


    # *********************
    # 着荷週が事業所nodeの非稼働週の場合 +1次週の着荷とする
    # *********************
    eta_shift = check_lv_week_fw(lv_week, eta_plan)  # ETD:Eatimate TimeDept.

    # リスト追加 extend
    # 安全在庫とカレンダ制約を考慮した着荷予定週Pに、w週Sからoffsetする

    # lot by lot operation
    # confirmed_P made by shifting parent_conf_S

    # ***********************
    # place_lot_supply_plan
    # ***********************


    # ここは、"REPLACE lot"するので、appendの前にchild psiをzero clearしてから

    # 今回のmodelでは、輸送工程もpsi nodeと同等に扱っている(=POではない)ので
    # 親のconfSを「そのままのWで」子のconfPに置く place_lotする
    child.psi4supply[w][3].append(lot)

    # 親のconfSを「輸送LT=0, 加工LT+SSでwをshiftして」子confSにplace_lotする
    child.psi4supply[eta_shift][0].append(lot)



def set_parent_all(node):
    # preordering

    if node.children == []:

        pass

    else:

        node.set_parent()  # この中で子nodeを見て親を教える。
        # def set_parent(self)

    for child in node.children:

        set_parent_all(child)


def print_parent_all(node):
    # preordering

    if node.children == []:

        pass

    else:

        print("node.parent and children", node.name, node.children)

    for child in node.children:

        print("child and parent", child.name, node.name)

        print_parent_all(child)


# def place_S_in_supply(child, lot): # lot SS shift on S


# 確定Pのセット

# replace lotするために、事前に、
# 出荷先となるすべてのchildren nodesのS[w][0]とP[w][3]をクリアしてplace lot
# ※出荷先ship2nodeを特定してからreplaceするのは難しいので、


def ship_lots2market(node, nodes):

    # キーが存在する場合は対応する値valueが返り、存在しない場合はNoneが返る。
    if node.children == []:  # 子nodeがないleaf nodeの場合

        pass

    else:

        # returnせずに子nodeのpsiのPに返す child.psi4demand[w][3]に直接セット
        # feedback_confirmedS2childrenP(node_req_plans, S_confirmed_plan)

        # ************************************
        # clearing children P[w][3] and S[w][0]
        # ************************************
        # replace lotするために、事前に、
        # 出荷先となるすべてのchildren nodesのS[w][0]とP[w][3]をクリア

        for child in node.children:

            for w in range(53 * node.plan_range):

                child.psi4supply[w][0] = []
                child.psi4supply[w][3] = []

        # lotidから、leaf_nodeを特定し、出荷先ship2nodeに出荷することは、
        # すべての子nodeに出荷することになる

        # ************************************
        # setting mother_confirmed_S
        # ************************************
        # このnode内での子nodeへの展開
        for w in range(53 * node.plan_range):


            # ある拠点の週次 生産出荷予定lots_list

            confirmed_S_lots = node.psi4supply[w][0]  # 親の確定出荷confS lot

            # 出荷先nodeを特定して

            # 一般には、下記のLT shiftだが・・・・・
            # 出荷先の ETA = LT_shift(ETD) でP place_lot
            # 工程中の ETA = SS_shift(ETD) でS place_lot

            # 本モデルでは、輸送工程 = modal_nodeを想定して・・・・・
            # 出荷先の ETA = 出荷元ETD        でP place_lot
            # 工程中の ETA = LT&SS_shift(ETD) でS place_lot
            # というイビツなモデル定義・・・・・

            # 直感的なPO=INVOICEという考え方に戻すべきかも・・・・・
            #
            # modal shiftのmodelingをLT_shiftとの拡張で考える???
            # modal = BOAT/AIR/QURIE
            # LT_shift(modal, w, ,,,,

            for lot in confirmed_S_lots:

                if lot == []:

                    pass

                else:


                    # lotidからleaf_nodeのpointerを返す
                    leaf_node_name = extract_node_name(lot)

                    leaf_node = nodes[leaf_node_name]

                    # 末端からあるnodeAまでleaf_nodeまでのnode_listをpathで返す

                    current_path = []
                    path = []

                    path = find_path_to_leaf_with_parent(node, leaf_node, current_path)

                    # nodes_listを逆にひっくり返す
                    path.reverse()

                    # あるnodeAから末端のleaf_nodeまでのnode_listをpathで返す
                    # path = find_path_to_leaf(node, leaf_node,current_path)


                    # 出荷先nodeはnodeAの次node、path[1]になる
                    ship2node_name = path[1]

                    ship2node = nodes[ship2node_name]

                    # ここでsupply planを更新している
                    # 出荷先nodeのPSIのPとSに、confirmed_S中のlotをby lotで置く
                    place_P_in_supply(w, ship2node, lot)

    for child in node.children:

        ship_lots2market(child, nodes)



def feedback_psi_lists(node, node_psi_dict, nodes):

    # キーが存在する場合は対応する値valueが返り、存在しない場合はNoneが返る。
    if node.children == []:  # 子nodeがないleaf nodeの場合

        pass

    else:

        # ************************************
        # clearing children P[w][3] and S[w][0]
        # ************************************
        # replace lotするために、事前に、
        # 出荷先となるすべてのchildren nodesのS[w][0]とP[w][3]をクリア

        for child in node.children:

            for w in range(53 * node.plan_range):

                child.psi4supply[w][0] = []
                child.psi4supply[w][3] = []

        # lotidから、leaf_nodeを特定し、出荷先ship2nodeに出荷することは、
        # すべての子nodeに出荷することになる

        # ************************************
        # setting mother_confirmed_S
        # ************************************
        # このnode内での子nodeへの展開
        for w in range(53 * node.plan_range):

            confirmed_S_lots = node.psi4supply[w][0]  # 親の確定出荷confS lot

            # 出荷先nodeを特定して

            # 一般には、下記のLT shiftだが・・・・・
            # 出荷先の ETA = LT_shift(ETD) でP place_lot
            # 工程中の ETA = SS_shift(ETD) でS place_lot

            # 本モデルでは、輸送工程 = modal_nodeを想定して・・・・・
            # 出荷先の ETA = 出荷元ETD        でP place_lot
            # 工程中の ETA = LT&SS_shift(ETD) でS place_lot
            # というイビツなモデル定義・・・・・

            # 直感的なPO=INVOICEという考え方に戻すべきかも・・・・・
            #
            # modal shiftのmodelingをLT_shiftとの拡張で考える???
            # modal = BOAT/AIR/QURIE
            # LT_shift(modal, w, ,,,,

            for lot in confirmed_S_lots:

                if lot == []:

                    pass

                else:

                    # *********************************************************
                    # child#ship2node = find_node_to_ship(node, lot)
                    # lotidからleaf_nodeのpointerを返す
                    leaf_node_name = extract_node_name(lot)

                    leaf_node = nodes[leaf_node_name]

                    # 末端からあるnodeAまでleaf_nodeまでのnode_listをpathで返す

                    current_path = []
                    path = []

                    path = find_path_to_leaf_with_parent(node, leaf_node, current_path)

                    # nodes_listを逆にひっくり返す
                    path.reverse()

                    # 出荷先nodeはnodeAの次node、path[1]になる
                    ship2node_name = path[1]

                    ship2node = nodes[ship2node_name]

                    # ここでsupply planを更新している
                    # 出荷先nodeのPSIのPとSに、confirmed_S中のlotをby lotで置く
                    place_P_in_supply(w, ship2node, lot)

    for child in node.children:

        feedback_psi_lists(child, node_psi_dict, nodes)



def get_all_psi4demand(node, node_all_psi):

    node_all_psi[node.name] = node.psi4demand

    for child in node.children:

        get_all_psi4demand(child, node_all_psi)

    return node_all_psi



def get_all_psi4demand_postorder(node, node_all_psi):

    node_all_psi[node.name] = node.psi4demand

    for child in node.children:

        get_all_psi4demand_postorder(child, node_all_psi)

    return node_all_psi



def get_all_psi4supply(node, node_all_psi):

    node_all_psi[node.name] = node.psi4supply

    for child in node.children:

        get_all_psi4supply(child, node_all_psi)

    return node_all_psi


def set_all_I4bullwhip(node):

    for child in node.children:

        set_all_I4bullwhip(child)

    # node辞書に時系列set
    # node.set_I4bullwhip()

    I_hi_len = []  # 在庫の高さ=リストの長さ

    for w in range(len(node.psi4demand)):

        I_hi_len.append(len(node.psi4demand[w][2]))

    node_I4bullwhip[node.name] = I_hi_len

    return node_I4bullwhip



def calc_all_psi2i4demand(node):

    # node_search.append(node)

    node.calcPS2I4demand()

    for child in node.children:

        calc_all_psi2i4demand(child)


def calcPS2I4demand2dict(node, node_psi_dict_In4Dm):

    plan_len = 53 * node.plan_range

    for w in range(1, plan_len):  # starting_I = 0 = w-1 / ending_I =plan_len

        s = node.psi4demand[w][0]
        co = node.psi4demand[w][1]

        i0 = node.psi4demand[w - 1][2]
        i1 = node.psi4demand[w][2]

        p = node.psi4demand[w][3]

        # *********************
        # # I(n-1)+P(n)-S(n)
        # *********************

        work = i0 + p  # 前週在庫と当週着荷分 availables

        # ここで、期末の在庫、S出荷=売上を操作している
        # S出荷=売上を明示的にlogにして、売上として記録し、表示する処理
        # 出荷されたS=売上、在庫I、未出荷COの集合を正しく表現する

        # モノがお金に代わる瞬間

        diff_list = [x for x in work if x not in s]  # I(n-1)+P(n)-S(n)

        node.psi4demand[w][2] = i1 = diff_list

    node_psi_dict_In4Dm[node.name] = node.psi4demand

    return node_psi_dict_In4Dm


# ********************
# inbound demand PS2I
# ********************


def calc_all_psi2i4demand_postorder(node, node_psi_dict_In4Dm):

    for child in node.children:

        calc_all_psi2i4demand_postorder(child, node_psi_dict_In4Dm)

    node_psi_dict_In4Dm = calcPS2I4demand2dict(node, node_psi_dict_In4Dm)

    node.psi4demand = node_psi_dict_In4Dm[node.name]  # 辞書をインスタンスに戻す


def calc_all_psi2i4supply(node):

    node.calcPS2I4supply()

    for child in node.children:

        calc_all_psi2i4supply(child)



def calc_all_psi2i_decouple4supply(
    node, nodes_decouple, decouple_flag, node_psi_dict_Ot4Dm, nodes_outbound
):

    # ********************************
    if node.name in nodes_decouple:

        decouple_flag = "ON"
    # ********************************

    if decouple_flag == "OFF":

        node.calcPS2I4supply()  # calc_psi with PUSH_S

    elif decouple_flag == "ON":

        # decouple nodeの場合は、psi処理後のsupply plan Sを出荷先nodeに展開する
        #
        # demand plan Sをsupply plan Sにcopyし、psi処理後に、supply plan Sを
        # PULL S / confirmed Sとして以降nodeのsupply planのSを更新する

        # ********************************

        if node.name in nodes_decouple:

            # 明示的に.copyする。
            plan_len = 53 * node.plan_range

            for w in range(0, plan_len):

                node.psi4supply[w][0] = node.psi4demand[w][0].copy()  

            node.calcPS2I4supply()  # calc_psi with PULL_S

            # *******************************************
            # decouple nodeは、pull_Sで出荷指示する
            # *******************************************
            ship_lots2market(node, nodes_outbound)

        else:

            #
            # decouple から先のnodeのpsi計算
            #

            # 明示的に.copyする。
            plan_len = 53 * node.plan_range

            for w in range(0, plan_len):

                node.psi4supply[w][0] = node.psi4demand[w][0].copy()  # @230728

            node.calcPS2I4supply()  # calc_psi with PULL_S

    else:

        print("error node decouple process " + node.name + " and " + nodes_decouple)

    for child in node.children:

        calc_all_psi2i_decouple4supply(
            child, nodes_decouple, decouple_flag, node_psi_dict_Ot4Dm, nodes_outbound
        )


# decople指定したnodeから先は、PULL情報で処理する
#
# decouple_flag node_decouple OFF JPN HAM
# decouple_flag node_decouple OFF TrBJPN2HAM HAM
# decouple_flag node_decouple ON HAM HAM
# decouple_flag node_decouple ON HAM_N HAM
# decouple_flag node_decouple ON HAM_D HAM
# decouple_flag node_decouple ON HAM_I HAM
# decouple_flag node_decouple ON MUC HAM
# decouple_flag node_decouple ON MUC_N HAM
# decouple_flag node_decouple ON MUC_D HAM
# decouple_flag node_decouple ON MUC_I HAM
# decouple_flag node_decouple ON FRALEAF HAM
# decouple_flag node_decouple OFF TrBJPN2SHA HAM
# decouple_flag node_decouple OFF SHA HAM
# decouple_flag node_decouple OFF SHA_N HAM
# decouple_flag node_decouple OFF SHA_D HAM
# decouple_flag node_decouple OFF SHA_I HAM
# decouple_flag node_decouple OFF TrBJPN2CAN HAM
# decouple_flag node_decouple OFF CAN HAM
# decouple_flag node_decouple OFF CAN_N HAM
# decouple_flag node_decouple OFF CAN_D HAM
# decouple_flag node_decouple OFF CAN_I HAM


def calc_all_psi2i_postorder(node):

    for child in node.children:

        calc_all_psi2i_postorder(child)

    node.calcPS2I4demand()  # backward plan with postordering



def calc_all_psiS2P_postorder(node):

    for child in node.children:

        calc_all_psiS2P_postorder(child)

    node.calcS2P()  # backward plan with postordering



# nodeを手繰りながらnode_psi_dict辞書を初期化する
def make_psi_space_dict(node, node_psi_dict, plan_range):

    psi_list = [[[] for j in range(4)] for w in range(53 * plan_range)]

    node_psi_dict[node.name] = psi_list  # 新しいdictにpsiをセット

    for child in node.children:

        make_psi_space_dict(child, node_psi_dict, plan_range)

    return node_psi_dict



# nodeを手繰りながらnode_psi_dict辞書を初期化する
def make_psi_space_zero_dict(node, node_psi_dict, plan_range):

    psi_list = [[0 for j in range(4)] for w in range(53 * plan_range)]

    node_psi_dict[node.name] = psi_list  # 新しいdictにpsiをセット

    for child in node.children:

        make_psi_space_zero_dict(child, node_psi_dict, plan_range)

    return node_psi_dict


# ****************************
# 辞書をinbound tree nodeのdemand listに接続する
# ****************************

def set_dict2tree_psi(node, attr_name, node_psi_dict):

    setattr(node, attr_name, node_psi_dict.get(node.name))

    # node.psi4supply = node_psi_dict.get(node.name)

    for child in node.children:

        set_dict2tree_psi(child, attr_name, node_psi_dict)


def set_dict2tree_InOt4AC(node, node_psi_dict):

    node.psi4accume = node_psi_dict.get(node.name)

    for child in node.children:

        set_dict2tree_InOt4AC(child, node_psi_dict)


def set_dict2tree_In4Dm(node, node_psi_dict):

    node.psi4demand = node_psi_dict.get(node.name)

    for child in node.children:

        set_dict2tree_In4Dm(child, node_psi_dict)


def set_dict2tree_In4Sp(node, node_psi_dict):

    node.psi4supply = node_psi_dict.get(node.name)

    for child in node.children:

        set_dict2tree_In4Sp(child, node_psi_dict)


def set_plan_range(node, plan_range):

    node.plan_range = plan_range

    for child in node.children:

        set_plan_range(child, plan_range)


# **********************************
# 多次元リストの要素数をcount
# **********************************
def multi_len(l):
    count = 0
    if isinstance(l, list):
        for v in l:
            count += multi_len(v)
        return count
    else:
        return 1


# a way of leveling
#
#      supply           demand
# ***********************************
# *                *                *
# * carry_over_out *                *
# *                *   S_lot        *
# *** capa_ceil ****   get_S_lot    *
# *                *                *
# *  S_confirmed   *                *
# *                *                *
# *                ******************
# *                *  carry_over_in *
# ***********************************

#
# carry_over_out = ( carry_over_in + S_lot ) - capa
#


def leveling_operation(carry_over_in, S_lot, capa_ceil):

    demand_side = []

    demand_side.extend(carry_over_in)

    demand_side.extend(S_lot)

    if len(demand_side) <= capa_ceil:

        S_confirmed = demand_side

        carry_over_out = []  # 繰り越し無し

    else:

        S_confirmed = demand_side[:capa_ceil]  # 能力内を確定する

        carry_over_out = demand_side[capa_ceil:]  # 能力を超えた分を繰り越す

    return S_confirmed, carry_over_out


# **************************
# leveling production
# **************************
def confirm_S(S_lots_list, prod_capa_limit, plan_range):

    S_confirm_list = [[] for i in range(53 * plan_range)]  # [[],[],,,,[]]

    carry_over_in = []

    week_no = 53 * plan_range - 1

    for w in range(week_no, -1, -1):  # 6,5,4,3,2,1,0

        S_lot = S_lots_list[w]
        capa_ceil = prod_capa_limit[w]

        S_confirmed, carry_over_out = leveling_operation(
            carry_over_in, S_lot, capa_ceil
        )

        carry_over_in = carry_over_out

        S_confirm_list[w] = S_confirmed

    return S_confirm_list

    # *********************************
    # visualise with 3D bar graph
    # *********************************


def show_inbound_demand(root_node_inbound):

    nodes_list, node_psI_list = extract_nodes_psI4demand(root_node_inbound)

    fig = visualise_psi_label(node_psI_list, nodes_list)

    offline.plot(fig, filename="inbound_demand_plan_010.html")


def connect_outbound2inbound(root_node_outbound, root_node_inbound):

    # ***************************************
    # setting root node OUTBOUND to INBOUND
    # ***************************************

    plan_range = root_node_outbound.plan_range

    for w in range(53 * plan_range):

        root_node_inbound.psi4demand[w][0] = root_node_outbound.psi4supply[w][0].copy()
        root_node_inbound.psi4demand[w][1] = root_node_outbound.psi4supply[w][1].copy()
        root_node_inbound.psi4demand[w][2] = root_node_outbound.psi4supply[w][2].copy()
        root_node_inbound.psi4demand[w][3] = root_node_outbound.psi4supply[w][3].copy()


        root_node_inbound.psi4supply[w][0] = root_node_outbound.psi4supply[w][0].copy()
        root_node_inbound.psi4supply[w][1] = root_node_outbound.psi4supply[w][1].copy()
        root_node_inbound.psi4supply[w][2] = root_node_outbound.psi4supply[w][2].copy()
        root_node_inbound.psi4supply[w][3] = root_node_outbound.psi4supply[w][3].copy()



#  class NodeのメソッドcalcS2Pと同じだが、node_psiの辞書を更新してreturn
def calc_bwd_inbound_si2p(node, node_psi_dict_In4Dm):

    # **************************
    # Safety Stock as LT shift
    # **************************
    # leadtimeとsafety_stock_weekは、ここでは同じ
    safety_stock_week = node.leadtime

    # **************************
    # long vacation weeks
    # **************************
    lv_week = node.long_vacation_weeks

    # S to P の計算処理  # dictに入れればself.psi4supplyから接続して見える
    node_psi_dict_In4Dm[node.name] = shiftS2P_LV(
        node.psi4demand, safety_stock_week, lv_week
    )

    return node_psi_dict_In4Dm


def calc_bwd_inbound_all_si2p(node, node_psi_dict_In4Dm):

    plan_range = node.plan_range

    # ********************************
    # inboundは、親nodeのSをそのままPに、shift S2Pして、node_spi_dictを更新
    # ********************************
    #    S2P # dictにlistセット
    node_psi_dict_In4Dm = calc_bwd_inbound_si2p(node, node_psi_dict_In4Dm)

    # *********************************
    # 子nodeがあればP2_child.S
    # *********************************

    if node.children == []:

        pass

    else:

        # inboundの場合には、dict=[]でセット済　代入する[]になる
        # 辞書のgetメソッドでキーnameから値listを取得。
        # キーが存在しない場合はNone
        # self.psi4demand = node_psi_dict_In4Dm.get(self.name)

        for child in node.children:

            for w in range(53 * plan_range):

                # move_lot P2S
                child.psi4demand[w][0] = node.psi4demand[w][3].copy()

    for child in node.children:

        calc_bwd_inbound_all_si2p(child, node_psi_dict_In4Dm)

    # stop 返さなくても、self.psi4demand[w][3]でPを参照できる。
    return node_psi_dict_In4Dm


# ************************
# sankey
# ************************
def make_outbound_sankey_nodes_preorder(
    week, node, nodes_all, all_source, all_target, all_value_acc
):

    for child in node.children:

        # 子nodeが特定したタイミングで親nodeと一緒にセット

        # source = node(from)のnodes_allのindexで返す
        # target = child(to)のnodes_allのindexで返す
        # value  = S: psi4supply[w][0]を取り出す

        all_source[week].append(nodes_all.index(str(node.name)))
        all_target[week].append(nodes_all.index(str(child.name)))

        if len(child.psi4demand[week][3]) == 0:

            work = 0  # dummy link
            # work = 0.1 # dummy link

        else:

            # child.をvalueとする
            work = len(child.psi4supply[week][3])

        value_acc = child.psi4accume[week][3] = child.psi4accume[week - 1][3] + work

        # accを[]にして、tree nodes listに戻してからvalueをセットする
        all_value_acc[week].append(value_acc)  # これも同じ辞書+リスト構造に

        make_outbound_sankey_nodes_preorder(
            week, child, nodes_all, all_source, all_target, all_value_acc
        )

    return all_source, all_target, all_value_acc


def make_inbound_sankey_nodes_postorder(
    week, node, nodes_all, all_source, all_target, all_value_acc
):

    for child in node.children:

        make_inbound_sankey_nodes_postorder(
            week, child, nodes_all, all_source, all_target, all_value_acc
        )

        # 子nodeが特定したタイミングで親nodeと一緒にセット

        # source = node(from)のnodes_allのindexで返す
        # target = child(to)のnodes_allのindexで返す
        # value  = S: psi4supply[w][0]を取り出す

        # ***********************
        # source_target_reverse
        # ***********************
        all_target[week].append(nodes_all.index(str(node.name)))
        all_source[week].append(nodes_all.index(str(child.name)))

        # all_source[week].append( nodes_all.index( str(node.name)  ) )
        # all_target[week].append( nodes_all.index( str(child.name) ) )

        if len(child.psi4demand[week][3]) == 0:

            # pass
            work = 0  # ==0でもlinkが見えるようにdummyで与える
            # work = 0.1  # ==0でもlinkが見えるようにdummyで与える

        else:

            # inboundのvalueは、子node数で割ることで親の数字と合わせる
            work = len(child.psi4demand[week][3]) / len(node.children)

        # @230610
        value_acc = child.psi4accume[week][3] = child.psi4accume[week - 1][3] + work

        all_value_acc[week].append(value_acc)
        # all_value[week].append( work )

        # all_value[week].append( len( child.psi4demand[week][3] ) )

    return all_source, all_target, all_value_acc

    # ********************************
    # end2end supply chain accumed plan
    # ********************************



def visualise_e2e_supply_chain_plan(root_node_outbound, root_node_inbound):

    # ************************
    # sankey
    # ************************

    nodes_outbound = []
    nodes_inbound = []
    node_psI_list = []

    nodes_outbound, node_psI_list = extract_nodes_psI4demand(root_node_outbound)

    nodes_inbound, node_psI_list = extract_nodes_psI4demand_postorder(root_node_inbound)

    nodes_all = []
    nodes_all = nodes_inbound + nodes_outbound[1:]

    all_source = {}  # [0,1,1,0,2,3,3] #sourceは出発元のnode
    all_target = {}  # [2,2,3,3,4,4,5] #targetは到着先のnode
    all_value = {}  # [8,1,3,2,9,3,2] #値
    all_value_acc = {}  # [8,1,3,2,9,3,2] #値

    plan_range = root_node_outbound.plan_range

    for week in range(1, plan_range * 53):

        all_source[week] = []
        all_target[week] = []
        all_value[week] = []
        all_value_acc[week] = []

        all_source, all_target, all_value_acc = make_outbound_sankey_nodes_preorder(
            week, root_node_outbound, nodes_all, all_source, all_target, all_value_acc
        )

        all_source, all_target, all_value_acc = make_inbound_sankey_nodes_postorder(
            week, root_node_inbound, nodes_all, all_source, all_target, all_value_acc
        )

    # init setting week
    week = 50

    data = dict(
        type="sankey",
        arrangement="fixed",  # node fixing option
        node=dict(
            pad=100,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes_all,  # 各nodeを作成
            # color = ["blue", "blue", "green", "green", "yellow", "yellow"] #色を指定します。
        ),
        link=dict(
            source=all_source[week],  # [0,1,1,0,2,3,3], #sourceは出発元のnode
            target=all_target[week],  # [2,2,3,3,4,4,5], #targetは到着先のnode
            value=all_value_acc[week]  # [8,1,3,2,9,3,2]   #流量
        ),
    )

    layout = dict(title="global weekly supply chain Sankey Diagram", font=dict(size=10))

    # **********************
    # frames 2 animation
    # **********************

    # フレームを保存するリスト
    frames = []

    ## プロットを保存するリスト
    # data = []
    # x = np.linspace(0, 1, 53*plan_range)

    # プロットの作成
    # 0, 0.1, ... , 5までのプロットを作成する
    # for step in np.linspace(0, 5, 51):

    week_len = 53 * plan_range

    # for step in np.linspace(0, week_len, week_len+1):

    for week in range(40, 53 * plan_range):

        frame_data = dict(
            type="sankey",
            arrangement="fixed",  # node fixing option
            node=dict(
                pad=100,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodes_all,  # 各nodeを作成
                ##color = ["blue", "blue", "green", "green", "yellow", "yellow"],
            ),
            link=dict(
                source=all_source[week],  # [0,1,1,0,2,3,3], #sourceは出発元のnode
                target=all_target[week],  # [2,2,3,3,4,4,5], #targetは到着先のnode
                value=all_value_acc[week],  # [8,1,3,2,9,3,2] #数量
            ),
        )

        frame_layout = dict(
            title="global weekly supply chain Week_No:" + str(week), font=dict(size=10)
        )

        frame = go.Frame(data=frame_data, layout=frame_layout)

        frames.append(frame)

        # ********************************
        # ココでpng出力
        # ********************************
        fig_temp = go.Figure(data=frame_data, layout=frame_layout)

        # ゼロ埋め
        # num = 12
        # f文字列：Python 3.6以降
        # s = f'{num:04}'  # 0埋めで4文字
        ##print(s)  # 0012

        zfill3_w = f"{week:03}"  # type is string

        temp_file_name = zfill3_w + ".png"

        pio.write_image(fig_temp, temp_file_name)  # write png

    fig = go.Figure(data=data, layout=layout, frames=frames)

    offline.plot(fig, filename="end2end_supply_chain_accumed_plan.html")



def map_psi_lots2df(node, psi_type, psi_lots):

    # preordering

    # psi4xxxx[w][0,1,2,3]で、node内のpsiをdfにcopy

    #    plan_len = 53 * node.plan_range
    #
    #    for w in range(1,plan_len):
    #
    #        s   = node.psi4demand[w][0]
    #        co  = node.psi4demand[w][1]
    #
    #        i0  = node.psi4demand[w-1][2]
    #        i1  = node.psi4demand[w][2]
    #
    #        p   = node.psi4demand[w][3]

    if psi_type == "demand":

        matrix = node.psi4demand

    elif psi_type == "supply":

        matrix = node.psi4supply

    else:

        print("error: wrong psi_type is defined")

    ## マッピングするデータのリスト
    #    psi_lots = []

    # マトリクスの各要素と位置をマッピング
    for week, row in enumerate(matrix):  # week

        for scoip, lots in enumerate(row):  # scoip

            for step_no, lot_id in enumerate(lots):

                psi_lots.append([node.name, week, scoip, step_no, lot_id])

    for child in node.children:

        map_psi_lots2df(child, psi_type, psi_lots)

    # DataFrameのカラム名
    # columns = ["step", "Element", "Position"]  # pos=(week,s-co-i-p)
    columns = ["node_name", "week", "s-co-i-p", "step_no", "lot_id"]

    # DataFrameの作成
    df = pd.DataFrame(psi_lots, columns=columns)

    return df



# *************************
# mapping psi tree2df    showing psi with plotly
# *************************
def show_psi_graph(root_node, D_S_flag, node_name, week_start, week_end):
    # def show_psi_graph(root_node_outbound,"demand","CAN_I",0,300):

    # show_psi_graph(
    #    root_node_outbound or root_node_inbound,  # out or in
    #    "demand"or "supply" ,                     # psi plan
    #    node_name,                                #"CAN_I" ,
    #    display_week_start,                       # 0 ,
    #    display_week_end,                         # 300 ,
    #    )

    # ********************************
    # map_psi_lots2df
    # ********************************

    # set_dataframe(root_node_outbound, root_node_inbound)

    if D_S_flag == "demand":

        psi_lots = []  # 空リストを持ってtreeの中に入る

        # tree中で、リストにpsiを入れ
        # DataFrameの作成して、dfを返している
        #     df = pd.DataFrame(psi_lots, columns=columns)

        df_demand_plan = map_psi_lots2df(root_node, D_S_flag, psi_lots)

    elif D_S_flag == "supply":

        psi_lots = []  # 空リストを持ってtreeの中に入る

        df_supply_plan = map_psi_lots2df(root_node, D_S_flag, psi_lots)

    else:

        print("error: combination  root_node==in/out  psi_plan==demand/supply")


    # **********************
    # select PSI
    # **********************

    if D_S_flag == "demand":

        df_init = df_demand_plan

    elif D_S_flag == "supply":

        df_init = df_supply_plan

    else:

        print("error: D_S_flag should be demand/sopply")

    # node指定
    node_show = node_name
    # node_show = "Platform"
    # node_show = "JPN"
    # node_show = "TrBJPN2HAM"
    # node_show = "HAM"
    # node_show = "MUC"
    # node_show = "MUC_D"
    # node_show = "SHA_D"
    # node_show = "SHA"
    # node_show = "CAN_I"

    ## 条件1: "node_name"の値がnode_show
    condition1 = df_init["node_name"] == node_show

    ## 条件2: "week"の値が50以上53以下
    # week_start, week_end

    condition2 = (df_init["week"] >= week_start) & (df_init["week"] <= week_end)
    # condition2 = (df_init["week"] >= 0) & (df_init["week"] <= 53 )
    # condition2 = (df_init["week"] >= 53) & (df_init["week"] <= 53+13 )
    # condition2 = (df_init["week"] >= 53) & (df_init["week"] <= 106)
    # condition2 = (df_init["week"] >= 0) & (df_init["week"] <= 300)

    ## 条件1のみでデータを抽出
    # df = df_init[condition1]

    ## 条件2のみでデータを抽出
    # df = df_init[condition2]

    ## 条件1と条件2のAND演算でデータを抽出
    df = df_init[condition1 & condition2]

    #    # 列名 "s-co-i-p" の値が 0 または 3 の行のみを抽出
    line_data_2I = df[df["s-co-i-p"].isin([2])]

    #    line_data_0 = df[df["s-co-i-p"].isin([0])]
    #    line_data_3 = df[df["s-co-i-p"].isin([3])]

    # 列名 "s-co-i-p" の値が 0 の行のみを抽出
    bar_data_0S = df[df["s-co-i-p"] == 0]

    # 列名 "s-co-i-p" の値が 3 の行のみを抽出
    bar_data_3P = df[df["s-co-i-p"] == 3]

    ## 列名 "s-co-i-p" の値が 2 の行のみを抽出
    # bar_data_2I = df[df["s-co-i-p"] == 2]

    # 折れ線グラフ用のデータを作成
    # 累積'cumsum'ではなく、'count'
    line_plot_data_2I = line_data_2I.groupby("week")["lot_id"].count()  ####.cumsum()

    #    line_plot_data_0 = line_data_0.groupby("week")["lot_id"].count().cumsum()
    #    line_plot_data_3 = line_data_3.groupby("week")["lot_id"].count().cumsum()

    # 積み上げ棒グラフ用のデータを作成
    bar_plot_data_3P = bar_data_3P.groupby("week")["lot_id"].count()
    bar_plot_data_0S = bar_data_0S.groupby("week")["lot_id"].count()


    # 積み上げ棒グラフのhovertemplate用のテキストデータを作成
    bar_hover_text_3P = (
        bar_data_3P.groupby("week")["lot_id"]
        .apply(lambda x: "<br>".join(x))
        .reset_index()
    )

    bar_hover_text_3P = bar_hover_text_3P["lot_id"].tolist()

    # 積み上げ棒グラフのhovertemplate用のテキストデータを作成
    bar_hover_text_0S = (
        bar_data_0S.groupby("week")["lot_id"]
        .apply(lambda x: "<br>".join(x))
        .reset_index()
    )
    bar_hover_text_0S = bar_hover_text_0S["lot_id"].tolist()

    # **************************
    # making graph
    # **************************
    # グラフの作成
    # fig = go.Figure()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    #    # 折れ線グラフの追加
    #    fig.add_trace(go.Scatter(x=line_plot_data_0.index,
    #                             y=line_plot_data_0.values,
    #                             mode='lines', name='Cumulative Count 0 S'),
    #        secondary_y=False )
    #
    #    fig.add_trace(go.Scatter(x=line_plot_data_3.index,
    #                             y=line_plot_data_3.values,
    #                             mode='lines', name='Cumulative Count 3 P'),
    #        secondary_y=False )

    # 積み上げ棒グラフの追加

    fig.add_trace(
        go.Bar(
            x=bar_plot_data_3P.index,
            y=bar_plot_data_3P.values,
            name="node 3_P: " + node_show,
            # name='Individual Count'+"3_P",
            text=bar_hover_text_3P,
            texttemplate="%{text}",
            textposition="inside",
            hovertemplate="Lot ID: %{x}<br>Count: %{y}",
        ),
        # hovertemplate='Lot ID: %{x}<br>Count: %{y}')
        # )
        secondary_y=False,
    )

    #    fig.add_trace(
    #
    #        go.Bar(      x=bar_plot_data_2I.index,
    #                     y=bar_plot_data_2I.values,
    #                     name='Individual Count'+"2_I",
    #
    #                     text=bar_hover_text_2I,
    #                     texttemplate='%{text}',
    #                     textposition= 'inside',
    #
    #                     #hovertemplate='Lot ID: %{x}<br>Count: %{y}'),
    #                     hovertemplate='Lot ID: %{x}<br>Count: %{y}')
    #                     )

    ##hovertemplate='Lot ID: %{text[x]}<br>Count: %{y}'),

    # secondary_y=True )

    fig.add_trace(
        go.Bar(
            x=bar_plot_data_0S.index,
            y=bar_plot_data_0S.values,
            name="node 0_S: " + node_show,
            # name='Individual Count'+"0_S",
            text=bar_hover_text_0S,
            texttemplate="%{text}",
            textposition="inside",
            hovertemplate="Lot ID: %{x}<br>Count: %{y}",
        ),
        # hovertemplate='Lot ID: %{x}<br>Count: %{y}')
        # )
        secondary_y=False,
    )

    # 折れ線グラフの追加
    fig.add_trace(
        go.Scatter(
            x=line_plot_data_2I.index,
            y=line_plot_data_2I.values,
            mode="lines",
            name="node 2_I: " + node_show,
        ),
        # name='Inventory 2I'),
        secondary_y=True,
    )

    # 軸ラベルの設定
    fig.update_xaxes(title_text="week")
    fig.update_yaxes(title_text="I by lots", secondary_y=True)

    # グラフの表示
    fig.show()


# *******************
# 生産平準化の前処理　ロット・カウント
# *******************
def count_lots_yyyy(psi_list, yyyy_str):

    matrix = psi_list

    # 共通の文字列をカウントするための変数を初期化
    count_common_string = 0

    # Step 1: マトリクス内の各要素の文字列をループで調べる
    for row in matrix:

        for element in row:

            # Step 2: 各要素内の文字列が "2023" を含むかどうかを判定
            if yyyy_str in element:

                # Step 3: 含む場合はカウンターを増やす
                count_common_string += 1

    return count_common_string


def is_52_or_53_week_year(year):
    # 指定された年の12月31日を取得
    last_day_of_year = datetime.date(year, 12, 31)

    # 12月31日のISO週番号を取得 (isocalendar()メソッドはタプルで[ISO年, ISO週番号, ISO曜日]を返す)
    _, iso_week, _ = last_day_of_year.isocalendar()

    # ISO週番号が1の場合は前年の最後の週なので、52週と判定
    if iso_week == 1:
        return 52
    else:
        return iso_week



def find_depth(node):
    if not node.parent:
        return 0
    else:
        return find_depth(node.parent) + 1


def find_all_leaves(node, leaves, depth=0):
    if not node.children:
        leaves.append((node, depth))  # (leafノード, 深さ) のタプルを追加
    else:
        for child in node.children:
            find_all_leaves(child, leaves, depth + 1)


def make_nodes_decouple_all(node):
    # def main():
    #
    #    root_node = build_tree()
    #    set_parent(root_node)

    #    leaves = []
    #    find_all_leaves(root_node, leaves)
    #    pickup_list = leaves[::-1]  # 階層の深い順に並べる

    leaves = []
    leaves_name = []

    nodes_decouple = []

    find_all_leaves(node, leaves)
    # find_all_leaves(root_node, leaves)
    pickup_list = sorted(leaves, key=lambda x: x[1], reverse=True)
    pickup_list = [leaf[0] for leaf in pickup_list]  # 深さ情報を取り除く

    # こうすることで、leaf nodeを階層の深い順に並べ替えた pickup_list が得られます。
    # 先に深さ情報を含めて並べ替え、最後に深さ情報を取り除くという流れになります。

    # 初期処理として、pickup_listをnodes_decoupleにcopy
    # pickup_listは使いまわしで、pop / insert or append / removeを繰り返す
    for nd in pickup_list:
        nodes_decouple.append(nd.name)

    nodes_decouple_all = []

    while len(pickup_list) > 0:

        # listのcopyを要素として追加
        nodes_decouple_all.append(nodes_decouple.copy())

        current_node = pickup_list.pop(0)
        del nodes_decouple[0]  # 並走するnode.nameの処理

        parent_node = current_node.parent

        if parent_node is None:
            break

        # 親ノードをpick up対象としてpickup_listに追加
        if current_node.parent:

            #    pickup_list.append(current_node.parent)
            #    nodes_decouple.append(current_node.parent.name)

            # if parent_node not in pickup_list:  # 重複追加を防ぐ

            # 親ノードの深さを見て、ソート順にpickup_listに追加
            depth = find_depth(parent_node)
            inserted = False

            for idx, node in enumerate(pickup_list):

                if find_depth(node) <= depth:

                    pickup_list.insert(idx, parent_node)
                    nodes_decouple.insert(idx, parent_node.name)

                    inserted = True
                    break

            if not inserted:
                pickup_list.append(parent_node)
                nodes_decouple.append(parent_node.name)

            # 親ノードから見た子ノードをpickup_listから削除
            for child in parent_node.children:

                if child in pickup_list:

                    pickup_list.remove(child)
                    nodes_decouple.remove(child.name)

        else:

            print("error: node dupplicated", parent_node.name)

    return nodes_decouple_all


def evaluate_inventory_all(node, total_I, node_eval_I, nodes_decouple):

    total_I_work = []

    # if node.name in nodes_decouple:

    # デカップル拠点nodeの在庫psi4xxx[w][2]をworkにappendする
    for w in range(node.plan_range * 53):
        total_I_work.append(len(node.psi4supply[w][2]))

    node_eval_I[node.name] = total_I_work

    # node.decoupling_total_I.extend( total_I_work )
    ##node.decoupling_total_I = total_I_work
    #
    # node_eval_I[node.name] = node.decoupling_total_I

    total_I += sum(total_I_work)  # sumをとる

    # デカップル拠点nodeのmax在庫をとる
    # max_I = max( max_I, max(total_I_work) ) # maxをとる

    # デカップル拠点nodeのmax在庫の累計をとる
    # total_I += max(total_I_work)

    # else:
    #
    #    pass

    if node.children == []:

        pass

    else:

        for child in node.children:

            total_I, node_eval_I = evaluate_inventory_all(
                child, total_I, node_eval_I, nodes_decouple
            )

    return total_I, node_eval_I


def evaluate_inventory(node, total_I, node_eval_I, nodes_decouple):

    total_I_work = []

    if node.name in nodes_decouple:

        # デカップル拠点nodeの在庫psi4xxx[w][2]をworkにappendする
        for w in range(node.plan_range * 53):
            total_I_work.append(len(node.psi4supply[w][2]))
            # total_I_work +=  len( node.psi4supply[w][2] )

        node_eval_I[node.name] = total_I_work

        # node.decoupling_total_I.extend( total_I_work )
        ##node.decoupling_total_I = total_I_work
        #
        # node_eval_I[node.name] = node.decoupling_total_I

        # デカップル拠点nodeのmax在庫の累計をとる
        total_I += max(total_I_work)
        # total_I = max( total_I, max(total_I_work) )

    else:

        pass

    if node.children == []:

        pass

    else:

        for child in node.children:

            total_I, node_eval_I = evaluate_inventory(
                child, total_I, node_eval_I, nodes_decouple
            )

    return total_I, node_eval_I


def show_subplots_set_y_axies(node_eval_I, nodes_decouple):

    nodes_decouple_text = ""

    for node_name in nodes_decouple:
        work_text = node_name + " "
        nodes_decouple_text += work_text

    # 各グラフのy軸の最大値を計算
    max_value = max(max(values) for values in node_eval_I.values())

    # サブプロットを作成
    fig = make_subplots(
        rows=len(node_eval_I),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[
            f"{key} (Max: {max(values)}, Sum: {sum(values)})"
            for key, values in node_eval_I.items()
        ],
    )

    # 各データをプロット
    row = 1
    for key, values in node_eval_I.items():

        max_sum_text = key + " max=" + str(max(values)) + " sum=" + str(sum(values))

        trace = go.Scatter(
            x=list(range(len(values))),
            y=values,
            fill="tozeroy",
            mode="none",
            name=max_sum_text,
        )

        # trace = go.Scatter(x=list(range(len(values))), y=values, fill='tozeroy', mode='none',  name = key )

        fig.add_trace(trace, row=row, col=1)
        row += 1

    # グラフのy軸の範囲を設定
    for i in range(1, len(node_eval_I) + 1):
        fig.update_yaxes(range=[0, max_value], row=i, col=1)

    # グラフレイアウトを設定
    fig.update_layout(
        title="デカップリング・ポイントの在庫推移" + nodes_decouple_text,
        # title='デカップリング・ポイントの在庫推移',
        xaxis_title="Week",
        yaxis_title="Lots",
        showlegend=False,  # 凡例を非表示
    )

    # グラフを表示
    fig.show()


def show_subplots_bar_decouple(node_eval_I, nodes_decouple):

    nodes_decouple_text = ""
    for node_name in nodes_decouple:
        work_text = node_name + " "
        nodes_decouple_text += work_text

    fig = make_subplots(
        rows=len(node_eval_I),
        cols=1,
        shared_xaxes=True,
        subplot_titles=list(node_eval_I.keys()),
    )

    row = 1
    for key, values in node_eval_I.items():
        trace = go.Scatter(
            x=list(range(len(values))), y=values, fill="tozeroy", mode="lines", name=key
        )
        fig.add_trace(trace, row=row, col=1)
        row += 1

    fig.update_layout(
        title="デカップリング・ポイントの在庫推移" + nodes_decouple_text,
        xaxis_title="week",
        yaxis_title="lots",
    )

    fig.show()


def show_subplots_bar(node_eval_I):

    fig = make_subplots(
        rows=len(node_eval_I),
        cols=1,
        shared_xaxes=True,
        subplot_titles=list(node_eval_I.keys()),
    )

    row = 1
    for key, values in node_eval_I.items():
        trace = go.Scatter(
            x=list(range(len(values))), y=values, fill="tozeroy", mode="lines", name=key
        )
        fig.add_trace(trace, row=row, col=1)
        row += 1

    fig.update_layout(title="デカップリング・ポイントの在庫推移", xaxis_title="week", yaxis_title="lots")

    fig.show()


# A = {
#    'NodeA': [10, 15, 8, 12, 20],
#    'NodeB': [5, 8, 6, 10, 12],
#    'NodeC': [2, 5, 3, 6, 8]
# }
#
# show_subplots_bar(A)


def show_node_eval_I(node_eval_I):

    ## サンプルの辞書A（キーがノード名、値が時系列データのリストと仮定）
    # A = {
    #    'NodeA': [10, 15, 8, 12, 20],
    #    'NodeB': [5, 8, 6, 10, 12],
    #    'NodeC': [2, 5, 3, 6, 8]
    # }

    # グラフ描画
    fig = px.line()
    for key, values in node_eval_I.items():
        fig.add_scatter(x=list(range(len(values))), y=values, mode="lines", name=key)

    fig.update_layout(title="デカップリング・ポイントの在庫推移", xaxis_title="week", yaxis_title="lots")

    fig.show()


# *******************************************
# 流動曲線で表示　show_flow_curve
# *******************************************


def show_flow_curve(df_init, node_show):

    # 条件1: "node_name"の値がnode_show
    condition1 = df_init["node_name"] == node_show

    ## 条件2: "week"の値が50以上53以下
    # condition2 = (df_init["week"] >= 0) & (df_init["week"] <= 53 )
    # condition2 = (df_init["week"] >= 53) & (df_init["week"] <= 53+13 )
    # condition2 = (df_init["week"] >= 53) & (df_init["week"] <= 106)

    # 条件1のみでデータを抽出
    df = df_init[condition1]

    ## 条件1と条件2のAND演算でデータを抽出
    # df = df_init[condition1 & condition2]

    # df_init = df_init[condition1 & condition2]
    # df_init = df_init[df_init['node_name']==node_show]

    # グループ化して小計"count"の計算
    df = df.groupby(["node_name", "week", "s-co-i-p"]).size().reset_index(name="count")

    # 累積値"count_accum"の計算
    df["count_accum"] = df.groupby(["node_name", "s-co-i-p"])["count"].cumsum()

    # 折れ線グラフの作成
    line_df_0 = df[df["s-co-i-p"].isin([0])]
    # s-co-i-pの値が0の行を抽出

    # 折れ線グラフの作成
    line_df_3 = df[df["s-co-i-p"].isin([3])]
    # s-co-i-pの値が3の行を抽出

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=line_df_0["week"],
            y=line_df_0["count_accum"],
            mode="lines",
            name="Demand S " + node_show,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=line_df_3["week"],
            y=line_df_3["count_accum"],
            mode="lines",
            name="Supply P " + node_show,
        ),
        secondary_y=False,
    )

    # 棒グラフの作成
    bar_df = df[df["s-co-i-p"] == 2]  # s-co-i-pの値が2の行を抽出

    fig.add_trace(
        go.Bar(x=bar_df["week"], y=bar_df["count"], name="Inventory "),
        # go.Bar(x=bar_df['week'], y=bar_df['step_no'], name='棒グラフ'),
        secondary_y=True,
    )

    # 軸ラベルの設定
    fig.update_xaxes(title_text="week")
    fig.update_yaxes(title_text="S and P", secondary_y=False)
    fig.update_yaxes(title_text="count_accum", secondary_y=True)
    # fig.update_yaxes(title_text='step_no', secondary_y=True)

    # グラフの表示
    fig.show()


# *******************************************
# tree handling parts
# *******************************************


def print_tree_bfs(root):
    queue = deque([(root, 0)])

    while queue:
        node, depth = queue.popleft()
        print("  " * depth + node.name)
        queue.extend((child, depth + 1) for child in node.children)


def print_tree_dfs(node, depth=0):
    print("  " * depth + node.name)
    for child in node.children:
        print_tree_dfs(child, depth + 1)


# *******************************************
# start main
# *******************************************


def main():

    # ***************************
    # build_psi_core
    # ***************************
    # build_psi_core()

    # create_tree

    # init_psi_dict # 辞書形式の3種類のpsiデータの初期化
    # demand plan, supply plan, decouple plan

    # init_set_psi_dict2tree # 辞書psiをtreeに初期セット

    # 以上で、psi_treeの骨格が完成
    # 後は、アプリケーション属性を追加して、操作と評価を繰り返す

    # ***************************
    # tree definition initialise
    # ***************************

    node_I4bullwhip = {}

    node_I4bullwhip = {}

    # ***************************
    # create outbound tree
    # ***************************

    # outbound_tree_file='supply_chain_tree_outbound_attributes_sssmall_JPN.csv'
    # outbound_tree_file = 'supply_chain_tree_outbound_attributes.csv'
    outbound_tree_file = "supply_chain_tree_outbound_attributes_JPN.csv"

    # nodes_xxxは、すべてのnodeインスタンスをnode_nameで引出せる辞書
    nodes_outbound = {}
    nodes_outbound, root_node_name = create_tree(outbound_tree_file)

    # rootのインスタンスを取得
    root_node_outbound = nodes_outbound[root_node_name]

    # root_node_outbound = nodes_outbound['JPN']
    # root_node_outbound = nodes_outbound['JPN_OUT']

    # 子nodeに親nodeをセットする
    set_parent_all(root_node_outbound)


    # ***************************
    # create inbound tree
    # ***************************

    inbound_tree_file = "supply_chain_tree_inbound_attributes_JPN.csv"

    nodes_inbound = {}

    nodes_inbound, root_node_name = create_tree(inbound_tree_file)

    root_node_inbound = nodes_inbound[root_node_name]


    # inboundの親子ホインタはセットしていない

    # ***************************
    # trans_month2week
    # ***************************

    # in_file    = "S_month_data.csv"
    # in_file    = "S_month_data_prev_year_sssmall_JPN.csv"

    in_file = "S_month_data_prev_year_JPN.csv"

    out_file = "S_iso_week_data.csv"

    plan_range = root_node_outbound.plan_range
    # plan_range = 1   #### 計画期間=1年

    node_yyyyww_value, node_yyyyww_key, plan_range, df_capa_year = trans_month2week(
        in_file, out_file
    )

    set_plan_range(root_node_outbound, plan_range)
    set_plan_range(root_node_inbound, plan_range)

    # an image of data
    #
    # for node_val in node_yyyyww_value:
    #   #print( node_val )
    #
    ##['SHA_N', 22.580645161290324, 22.580645161290324, 22.580645161290324, 22.5    80645161290324, 26.22914349276974, 28.96551724137931, 28.96551724137931, 28.    96551724137931, 31.067853170189103, 33.87096774193549, 33.87096774193549, 33    .87096774193549, 33.87096774193549, 30.33333333333333, 30.33333333333333, 30    .33333333333333, 30.33333333333333, 31.247311827956988, 31.612903225806452,

    # node_yyyyww_key [['CAN', 'CAN202401', 'CAN202402', 'CAN202403', 'CAN20240    4', 'CAN202405', 'CAN202406', 'CAN202407', 'CAN202408', 'CAN202409', 'CAN202    410', 'CAN202411', 'CAN202412', 'CAN202413', 'CAN202414', 'CAN202415', 'CAN2    02416', 'CAN202417', 'CAN202418', 'CAN202419',

    # ********************************
    # make_node_psi_dict
    # ********************************
    # 1. treeを生成して、nodes[node_name]辞書で、各nodeのinstanceを操作する
    # 2. 週次S yyyywwの値valueを月次Sから変換、
    #    週次のlotの数Slotとlot_keyを生成、
    # 3. ロット単位=lot_idとするリストSlot_id_listを生成しながらpsi_list生成
    # 4. node_psi_dict=[node1: psi_list1,,,]を生成、treeのnode.psi4demandに接続する

    S_week = []

    # *************************************************
    # initialise node_psi_dict
    # *************************************************
    node_psi_dict = {}  # 変数 node_psi辞書

    # ***************************
    # outbound psi_dic
    # ***************************
    node_psi_dict_Ot4Dm = {}  # node_psi辞書4demand plan
    node_psi_dict_Ot4Sp = {}  # node_psi辞書4supply plan

    # coupling psi
    node_psi_dict_Ot4Cl = {}  # node_psi辞書4couple plan

    # accume psi
    node_psi_dict_Ot4Ac = {}  # node_psi辞書outbound4accume plan

    # ***************************
    # inbound psi_dic
    # ***************************
    node_psi_dict_In4Dm = {}  # node_psi辞書inbound4demand plan
    node_psi_dict_In4Sp = {}  # node_psi辞書inbound4supply plan

    # coupling psi
    node_psi_dict_In4Cl = {}  # node_psi辞書inbound4couple plan

    # accume psi
    node_psi_dict_In4Ac = {}  # node_psi辞書inbound4accume plan

    # rootからtree nodeをpreorder順に検索 node_psi辞書を空リストを作る
    node_psi_dict_Ot4Dm = make_psi_space_dict(
        root_node_outbound, node_psi_dict_Ot4Dm, plan_range
    )
    node_psi_dict_Ot4Sp = make_psi_space_dict(
        root_node_outbound, node_psi_dict_Ot4Sp, plan_range
    )
    node_psi_dict_Ot4Cl = make_psi_space_dict(
        root_node_outbound, node_psi_dict_Ot4Cl, plan_range
    )
    node_psi_dict_Ot4Ac = make_psi_space_dict(
        root_node_outbound, node_psi_dict_Ot4Ac, plan_range
    )

    node_psi_dict_In4Dm = make_psi_space_dict(
        root_node_inbound, node_psi_dict_In4Dm, plan_range
    )
    node_psi_dict_In4Sp = make_psi_space_dict(
        root_node_inbound, node_psi_dict_In4Sp, plan_range
    )
    node_psi_dict_In4Cl = make_psi_space_dict(
        root_node_inbound, node_psi_dict_In4Cl, plan_range
    )
    node_psi_dict_In4Ac = make_psi_space_dict(
        root_node_inbound, node_psi_dict_In4Ac, plan_range
    )

    # ***********************************
    # set_dict2tree
    # ***********************************
    # rootからtreeをpreorder順に検索 node_psi辞書をnodeにset
    set_dict2tree_psi(root_node_outbound, "psi4demand", node_psi_dict_Ot4Dm)
    set_dict2tree_psi(root_node_outbound, "psi4supply", node_psi_dict_Ot4Sp)
    set_dict2tree_psi(root_node_outbound, "psi4couple", node_psi_dict_Ot4Cl)
    set_dict2tree_psi(root_node_outbound, "psi4accume", node_psi_dict_Ot4Ac)

    set_dict2tree_psi(root_node_inbound, "psi4demand", node_psi_dict_In4Dm)
    set_dict2tree_psi(root_node_inbound, "psi4supply", node_psi_dict_In4Sp)
    set_dict2tree_psi(root_node_inbound, "psi4couple", node_psi_dict_In4Cl)
    set_dict2tree_psi(root_node_inbound, "psi4accume", node_psi_dict_In4Ac)


    # *********************************
    # inbound data initial setting
    # *********************************

    # *********************************
    # node_psi辞書を作成して、node.psiにセットする
    # *********************************
    node_psi_dict_In4Dm = {}  # node_psi辞書を定義 # Inbound for Demand
    node_psi_dict_In4Sp = {}  # node_psi辞書を定義 # Inbound for Supply

    # rootからtree nodeをinbound4demand=preorder順に検索 node_psi辞書をmake
    node_psi_dict_In4Dm = make_psi_space_dict(
        root_node_inbound, node_psi_dict_In4Dm, plan_range
    )
    node_psi_dict_In4Sp = make_psi_space_dict(
        root_node_inbound, node_psi_dict_In4Sp, plan_range
    )

    # ***********************************
    # set_dict2tree
    # ***********************************
    # rootからtreeをinbound4demand=preorder順に検索 node_psi辞書をnodeにset
    set_dict2tree_In4Dm(root_node_inbound, node_psi_dict_In4Dm)
    set_dict2tree_In4Sp(root_node_inbound, node_psi_dict_In4Sp)

    # node_nameを先頭に、ISO week順にSのリストで持つ
    # leaf_nodeにはISO week Sが入っているが、
    # leaf以外のnode値=0 (需要シフト時に生成される)

    S_lots_dict = make_S_lots(node_yyyyww_value, node_yyyyww_key, nodes_outbound)

    set_Slots2psi4demand(root_node_outbound, S_lots_dict)

    show_sw = 0  # 1 or 0

    # node指定
    # node_show = "Platform"
    # node_show = "JPN"
    # node_show = "TrBJPN2HAM"

    node_show = "HAM"

    # node_show = "MUC"
    # node_show = "MUC_D"
    # node_show = "SHA"
    # node_show = "SHA_D"
    # node_show = "CAN_I"

    if show_sw == 1:
        show_psi_graph(root_node_outbound, "demand", node_show, 0, 300)

    # demand planのS Pの生成

    # ***************************************
    # you can see root_node_outbound with "mplot3d" if you want
    # ****************************************
    # show_psi_3D_graph_node(root_node_outbound)

    # ***************************************
    # calc_all_psi2i
    # ***************************************
    # SP2I計算はpreorderingでForeward     Planningする
    calc_all_psi2i4demand(root_node_outbound)

    if show_sw == 1:
        show_psi_graph(root_node_outbound, "demand", node_show, 0, 300)


    # *********************************
    # mother plant capacity parameter
    # *********************************

    demand_supply_ratio = 3  # demand_supply_ratio = ttl_supply / ttl_demand

    # ********************
    # common_plan_unit_lot_size
    # OR
    # lot_size on root( = mother plant )
    # ********************
    plant_lot_size = 0

    # mother plantのlot_size定義を取るのはやめて、
    # common plant unitとして一つのlot_sizeを使う

    common_plan_unit_lot_size = 1  # 100 #24 #50 # 100  # 100   # 3 , 10, etc
    # common_plan_unit_lot_size = 100 #24 #50 # 100  # 100   # 3 , 10, etc

    plant_lot_size = common_plan_unit_lot_size

    # plant_lot_size     = root_node_outbound.lot_size # parameter master file


    # ********************
    # 辞書 year key: total_demand
    # ********************

    # 切り捨ては、a//b
    # 切り上げは、(a+b-1)//b

    plant_capa_vol = {}
    plant_capa_lot = {}

    week_vol = 0

    for i, row in df_capa_year.iterrows():

        plant_capa_vol[row["year"]] = row["total_demand"]

        # plant_capa_lot[row['year']] = (row['total_demand']+plant_lot_size -1)//     plant_lot_size # 切り上げ

        week_vol = row["total_demand"] * demand_supply_ratio // 52

        plant_capa_lot[row["year"]] = (week_vol + plant_lot_size - 1) // plant_lot_size

        # plant_capa_lot[row['year']] = ((row['total_demand']+52-1 // 52)+plant_lot_size-1) // plant_lot_size
        # plant_capa_lot[row['year']] = row['total_demand'] // plant_lot_size

    # **********************
    # ISO weekが年によって52と53がある
    # ここでは、53*plan_rangeの年別53週のaverage_capaとして定義
    # **********************

    # 53*plan_range
    #

    year_st = 2020
    year_end = 2021

    year_st = df_capa_year["year"].min()
    year_end = df_capa_year["year"].max()

    week_capa = []
    week_capa_w = []

    for year in range(year_st, year_end + 1):  # 5_years

        week_capa_w = [plant_capa_lot[year]] * 53
        # week_capa_w = [ (plant_capa_lot[year] + 53 - 1) // 53 ] * 53

        week_capa += week_capa_w

    leveling_S_in = []

    leveling_S_in = root_node_outbound.psi4demand



    # calendar　先行生産によるキャパ対応、

    # *****************************
    # mother plan leveling    setting initial data
    # *****************************

    # a sample data setting

    week_no = 53 * plan_range

    S_confirm = 15

    S_lots = []
    S_lots_list = []

    for w in range(53 * plan_range):

        S_lots_list.append(leveling_S_in[w][0])

    prod_capa_limit = week_capa

    # ******************
    # initial setting
    # ******************

    capa_ceil = 50
    # capa_ceil = 100
    # capa_ceil = 10

    S_confirm_list = confirm_S(S_lots_list, prod_capa_limit, plan_range)

    # **********************************
    # 多次元リストの要素数をcountして、confirm処理の前後の要素数を比較check
    # **********************************
    S_lots_list_element = multi_len(S_lots_list)

    S_confirm_list_element = multi_len(S_confirm_list)

    # *********************************
    # initial setting
    # *********************************
    node_psi_dict_Ot4Sp = {}  # node_psi_dict_Ot4Spの初期セット

    node_psi_dict_Ot4Sp = make_psi4supply(root_node_outbound, node_psi_dict_Ot4Sp)

    #
    # node_psi_dict_Ot4Dmでは、末端市場のleafnodeのみセット
    #
    # root_nodeのS psi_list[w][0]に、levelingされた確定出荷S_confirm_listをセッ    ト

    # 年間の総需要(総lots)をN週先行で生産する。
    # 例えば、３ヶ月先行は13週先行生産として、年間総需要を週平均にする。

    # S出荷で平準化して、confirmedS-I-P
    # conf_Sからconf_Pを生成して、conf_P-S-I  PUSH and PULL

    S_list = []
    S_allocated = []

    year_lots_list = []
    year_week_list = []

    leveling_S_in = []

    leveling_S_in = root_node_outbound.psi4demand

    # psi_listからS_listを生成する
    for psi in leveling_S_in:

        S_list.append(psi[0])

    # 開始年を取得する
    plan_year_st = year_st  # 開始年のセット in main()要修正

    for yyyy in range(plan_year_st, plan_year_st + plan_range + 1):

        year_lots = count_lots_yyyy(S_list, str(yyyy))

        year_lots_list.append(year_lots)

    #        # 結果を出力
    #       #print(yyyy, " year carrying lots:", year_lots)
    #
    #    # 結果を出力
    #   #print(" year_lots_list:", year_lots_list)

    # an image of sample data
    #
    # 2023  year carrying lots: 0
    # 2024  year carrying lots: 2919
    # 2025  year carrying lots: 2914
    # 2026  year carrying lots: 2986
    # 2027  year carrying lots: 2942
    # 2028  year carrying lots: 2913
    # 2029  year carrying lots: 0
    #
    # year_lots_list: [0, 2919, 2914, 2986, 2942, 2913, 0]

    year_list = []

    for yyyy in range(plan_year_st, plan_year_st + plan_range + 1):

        year_list.append(yyyy)

        # テスト用の年を指定
        year_to_check = yyyy

        # 指定された年のISO週数を取得
        week_count = is_52_or_53_week_year(year_to_check)

        year_week_list.append(week_count)

    #        # 結果を出力
    #       #print(year_to_check, " year has week_count:", week_count)
    #
    #    # 結果を出力
    #   #print(" year_week_list:", year_week_list)

   #print("year_list", year_list)

    # an image of sample data
    #
    # 2023  year has week_count: 52
    # 2024  year has week_count: 52
    # 2025  year has week_count: 52
    # 2026  year has week_count: 53
    # 2027  year has week_count: 52
    # 2028  year has week_count: 52
    # 2029  year has week_count: 52
    # year_week_list: [52, 52, 52, 53, 52, 52, 52]

    # *****************************
    # 生産平準化のための年間の週平均生産量(ロット数単位)
    # *****************************

    # *****************************
    # make_year_average_lots
    # *****************************
    # year_list     = [2023,2024,2025,2026,2027,2028,2029]

    # year_lots_list = [0, 2919, 2914, 2986, 2942, 2913, 0]
    # year_week_list = [52, 52, 52, 53, 52, 52, 52]

    year_average_lots_list = []

    for lots, weeks in zip(year_lots_list, year_week_list):
        average_lots_per_week = math.ceil(lots / weeks)
        year_average_lots_list.append(average_lots_per_week)

    #print("year_average_lots_list", year_average_lots_list)
    #
    # an image of sample data
    #
    # year_average_lots_list [0, 57, 57, 57, 57, 57, 0]

    # 年間の総需要(総lots)をN週先行で生産する。
    # 例えば、３ヶ月先行は13週先行生産として、年間総需要を週平均にする。

    #
    # 入力データの前提
    #
    # leveling_S_in[w][0] == S_listは、outboundのdemand_planで、
    # マザープラントの出荷ポジションのSで、
    # 5年分 週次 最終市場におけるlot_idリストが
    # LT offsetされた状態で入っている
    #
    # year_list     = [2023,2024,2025,2026,2027,2028,2029]

    # year_lots_list = [0, 2919, 2914, 2986, 2942, 2913, 0]
    # year_week_list = [52, 52, 52, 53, 52, 52, 52]
    # year_average_lots_list [0, 57, 57, 57, 57, 57, 0]

    # ********************************
    # 先行生産の週数
    # ********************************
    # precedence_production_week =13

    # pre_prod_week =26 # 26週=6か月の先行生産をセット
    # pre_prod_week =13 # 13週=3か月の先行生産をセット
    pre_prod_week = 6  # 6週=1.5か月の先行生産をセット

    # ********************************
    # 先行生産の開始週を求める
    # ********************************
    # 市場投入の前年において i= 0  year_list[i]           # 2023
    # 市場投入の前年のISO週の数 year_week_list[i]         # 52

    # 先行生産の開始週は、市場投入の前年のISO週の数 - 先行生産週

    pre_prod_start_week = 0

    i = 0

    pre_prod_start_week = year_week_list[i] - pre_prod_week

    # スタート週の前週まで、[]リストで埋めておく
    for i in range(pre_prod_start_week):
        S_allocated.append([])

    # ********************************
    # 最終市場からのLT offsetされた出荷要求lot_idリストを
    # Allocate demand to mother plant weekly slots
    # ********************************

    # S_listの週別lot_idリストを一直線のlot_idリストに変換する
    # mother plant weekly slots

    # 空リストを無視して、一直線のlot_idリストに変換

    # 空リストを除外して一つのリストに結合する処理
    S_one_list = [item for sublist in S_list if sublist for item in sublist]

    ## 結果表示
    ##print(S_one_list)

    # to be defined 毎年の定数でのlot_idの切り出し

    # listBの各要素で指定された数だけlistAから要素を切り出して
    # 新しいリストlistCを作成

    listA = S_one_list  # 5年分のlot_idリスト

    listB = year_lots_list  # 毎年毎の総ロット数

    listC = []  # 毎年のlot_idリスト

    start_idx = 0

    for i, num in enumerate(listB):

        end_idx = start_idx + num

        # original sample
        # listC.append(listA[start_idx:end_idx])

        # **********************************
        # "slice" and "allocate" at once
        # **********************************
        sliced_lots = listA[start_idx:end_idx]

        # 毎週の生産枠は、year_average_lots_listの平均値を取得する。
        N = year_average_lots_list[i]

        if N == 0:

            pass

        else:

            # その年の週次の出荷予定数が生成される。
            S_alloc_a_year = [
                sliced_lots[j : j + N] for j in range(0, len(sliced_lots), N)
            ]

            S_allocated.extend(S_alloc_a_year)
            # S_allocated.append(S_alloc_a_year)

        start_idx = end_idx

    ## 結果表示
    #print("S_allocated", S_allocated)

    # set psi on outbound supply

    # "JPN-OUT"
    #

    node_name = root_node_outbound.name  # Nodeからnode_nameを取出す

    # for w, pSi in enumerate( S_allocated ):
    #
    #    node_psi_dict_Ot4Sp[node_name][w][0] = pSi

    for w in range(53 * plan_range):

        if w <= len(S_allocated) - 1:  # index=0 start

            node_psi_dict_Ot4Sp[node_name][w][0] = S_allocated[w]

        else:

            node_psi_dict_Ot4Sp[node_name][w][0] = []

    # supply_plan用のnode_psi_dictをtree構造のNodeに接続する
    # Sをnode.psi4supplyにset  # psi_listをclass Nodeに接続

    set_psi_lists4supply(root_node_outbound, node_psi_dict_Ot4Sp)

    # この後、
    # process_1 : mother plantとして、S2I_fixed2Pで、出荷から生産を確定
    # process_2 : 子nodeに、calc_S2P2psI()でfeedbackする。



    #if show_sw == 1:
    #    show_psi_graph(root_node_outbound, "supply", node_show, 0, 300)



    # demand planからsupply planの初期状態を生成

    # *************************
    # process_1 : mother plantとして、S2I_fixed2Pで、出荷から生産を確定
    # *************************

    # calcS2fixedI2P
    # psi4supplyを対象にする。
    # psi操作の結果Pは、S2PをextendでなくS2Pでreplaceする

    root_node_outbound.calcS2P_4supply()  # mother plantのconfirm S=> P

    root_node_outbound.calcPS2I4supply()  # mother plantのPS=>I

    # show_psi_graph(root_node_outbound,"supply", node_show, 0, 300 )
    if show_sw == 1:
        show_psi_graph(root_node_outbound, "supply", "JPN", 0, 300)

    # mother plantのみのleveling生産平準化処理
    # mother plant="JPN"のsupply planを、年間需要の週平均値でlot数を平均化

    # *************************
    # process_2 : 子nodeに、calc_S2P2psI()でfeedbackする。
    # *************************

    #
    # 接続して、mother_plant確定Sを生成し、tree中の子nodeに確定lotをfeedback
    #

    ##print('feedback node_psi_dict_Ot4Sp',node_psi_dict_Ot4Sp)

    # ***************************************
    # その3　都度のparent searchを実行
    # ***************************************
    feedback_psi_lists(root_node_outbound, node_psi_dict_Ot4Sp, nodes_outbound)

    #
    # lot by lotのサーチ　その1 遅い
    #
    # ***************************************
    # S_confirm_list: mother planの出荷計画を平準化、確定した出荷計画を
    # children_node.P_request  : すべての子nodeの出荷要求数のリストと比較して、
    # children_node.P_confirmed: それぞれの子nodeの出荷確定数を生成する
    # ***************************************

    #
    # lot by lotのサーチ その2 少し遅い
    #
    # ***************************************
    # tree上の子nodeをサーチして、子nodeのSに、lotかあるか(=出荷先nodeか)
    # ***************************************

    #
    # lot by lotのサーチ その3 少し早い
    #
    # ***************************************
    # lot処理の都度、以下のサーチを実行
    # lot_idが持つleaf_nodeの情報から、parent_nodeをサーチ、出荷先nodeを特定
    # ***************************************

    # lot by lotのサーチ その4 早いハズ
    #
    # ***************************************
    # creat_treeの後、leaf_nodeの辞書に、reverse(leaf_root_list)を作り、
    # lot_idが持つleaf_node情報から、leaf_root辞書から出荷先nodeを特定
    # root_leaf_list中の「指定したnodeの次」list[index(node)+1]を取り出す
    # ***************************************


    #if show_sw == 1:
    #    show_psi_graph(root_node_outbound, "supply", node_show, 0, 300)


    # *********************************
    # make visualise data for 3D bar graph
    # *********************************
    #    visualise_inventory4demand_3d_bar(root_node_outbound, 'demand_I_bar.html')


    # ***************************************
    # decouple nodeを判定して、
    # calc_all_psi2iのSをPUSHからPULLに切り替える
    # ***************************************

    # nodes_decouple_all = [] # nodes_decoupleのすべてのパターンをリストアップ
    #
    # decoupleパターンを計算・評価
    # for i, nodes_decouple in enumerate(nodes_decouple_all):
    #    calc_xxx(root_node, , , , )
    #    eval_xxx(root_node, eval)

    nodes_decouple_all = make_nodes_decouple_all(root_node_outbound)

    #print("nodes_decouple_all", nodes_decouple_all)
    #
    # nodes_decouple_all [
    # ['JPN'],
    # ['TrBJPN2HAM', 'TrBJPN2SHA', 'TrBJPN2CAN'],
    # ['TrBJPN2SHA', 'TrBJPN2CAN', 'HAM'],
    # ['TrBJPN2CAN', 'HAM', 'SHA'],
    # ['HAM', 'SHA', 'CAN'],
    # ['SHA', 'CAN', 'HAM_N', 'HAM_D', 'HAM_I', 'MUC', 'FRALEAF'],
    # ['CAN', 'HAM_N', 'HAM_D', 'HAM_I', 'MUC', 'FRALEAF', 'SHA_N', 'SHA_D', 'SHA_I'],
    # ['HAM_N', 'HAM_D', 'HAM_I', 'MUC', 'FRALEAF', 'SHA_N', 'SHA_D', 'SHA_I', 'CAN_N', 'CAN_D', 'CAN_I'],
    # ['HAM_D', 'HAM_I', 'MUC', 'FRALEAF', 'SHA_N', 'SHA_D', 'SHA_I', 'CAN_N', 'CAN_D', 'CAN_I'],
    # ['HAM_I', 'MUC', 'FRALEAF', 'SHA_N', 'SHA_D', 'SHA_I', 'CAN_N', 'CAN_D', 'CAN_I'],

    # ['MUC', 'FRALEAF', 'SHA_N', 'SHA_D', 'SHA_I', 'CAN_N', 'CAN_D', 'CAN_I'],

    # ['FRALEAF', 'SHA_N', 'SHA_D', 'SHA_I', 'CAN_N', 'CAN_D', 'CAN_I', 'MUC_N', 'MUC_D', 'MUC_I'], ['SHA_N', 'SHA_D', 'SHA_I', 'CAN_N', 'CAN_D', 'CAN_I', 'MUC_N', 'MUC_D', 'MUC_I'], ['SHA_D', 'SHA_I', 'CAN_N', 'CAN_D', 'CAN_I', 'MUC_N', 'MUC_D', 'MUC_I'], ['SHA_I', 'CAN_N', 'CAN_D', 'CAN_I', 'MUC_N', 'MUC_D', 'MUC_I'], ['CAN_N', 'CAN_D', 'CAN_I', 'MUC_N', 'MUC_D', 'MUC_I'], ['CAN_D', 'CAN_I', 'MUC_N', 'MUC_D', 'MUC_I'], ['CAN_I', 'MUC_N', 'MUC_D', 'MUC_I'], ['MUC_N', 'MUC_D', 'MUC_I'], ['MUC_D', 'MUC_I'], ['MUC_I']]

    # SP2I4supplyの計算はsupply planのpsiをpreorderingでForeward Planningする

    for i, nodes_decouple in enumerate(nodes_decouple_all):

        decouple_flag = "OFF"

        calc_all_psi2i_decouple4supply(
            root_node_outbound,
            nodes_decouple,
            decouple_flag,
            node_psi_dict_Ot4Dm,
            nodes_outbound,
        )

        # outbound supplyのIをsummary
        # setting on "node.decoupling_total_I"

        total_I = 0

        node_eval_I = {}

        # decoupleだけでなく、tree all nodeでグラフ表示
        total_I, node_eval_I = evaluate_inventory_all(
            root_node_outbound, total_I, node_eval_I, nodes_decouple
        )

        show_subplots_set_y_axies(node_eval_I, nodes_decouple)


    # show_psi_graph(root_node_outbound,"demand", node_show, 0, 300 ) #
    # show_psi_graph(root_node_outbound,"supply", node_show, 0, 300 ) #

    if show_sw == 2:

        show_psi_graph(root_node_outbound, "supply", "JPN", 0, 300)  #
        show_psi_graph(root_node_outbound, "supply", "TrBJPN2HAM", 0, 300)  #
        show_psi_graph(root_node_outbound, "supply", "HAM", 0, 300)  #
        show_psi_graph(root_node_outbound, "supply", "MUC", 0, 300)  #
        show_psi_graph(root_node_outbound, "supply", "MUC_D", 0, 300)  #

        show_psi_graph(root_node_outbound, "supply", "HAM_D", 0, 300)  #
        show_psi_graph(root_node_outbound, "supply", "HAM_I", 0, 300)  #
        show_psi_graph(root_node_outbound, "supply", "HAM_N", 0, 300)  #

    #show_psi_graph(root_node_outbound, "supply", "JPN", 0, 300)  #

    # *********************************
    # make visualise data for 3D bar graph
    # *********************************
    #    visualise_inventory4supply_3d_bar(root_node_outbound, 'supply_I_bar.html')

    # *********************************
    # psi4accume  accume_psi initial setting on Inbound and Outbound
    # *********************************

    # *********************************
    # node_psi辞書を作成して、node.psiにセットする
    # *********************************
    node_psi_dict_In4Ac = {}  # node_psi辞書を定義 # Inbound for Accume
    node_psi_dict_Ot4Ac = {}  # node_psi辞書を定義 # Outbound for Accume

    # *********************************
    # make dict from tree getting node_name and setting [[]*53*plan_range]
    # *********************************
    # inboundとoutboundのtreeをrootからpreorder順に検索 node_psi辞書をmake

    node_psi_dict_Ot4Ac = make_psi_space_zero_dict(
        root_node_outbound, node_psi_dict_Ot4Ac, plan_range
    )

    node_psi_dict_In4Ac = make_psi_space_zero_dict(
        root_node_inbound, node_psi_dict_In4Ac, plan_range
    )

    # ***********************************
    # set_dict2tree
    # ***********************************
    # rootから in&out treeをpreorder順に検索 node_psi辞書をnodeにset

    # psi4accumeは、inbound outbound共通
    set_dict2tree_InOt4AC(root_node_outbound, node_psi_dict_Ot4Ac)
    set_dict2tree_InOt4AC(root_node_inbound, node_psi_dict_In4Ac)

    # class Nodeのnode.psi4accumeにセット
    # node.psi4accume = node_psi_dict.get(node.name)

    # *********************************
    # inbound data initial setting
    # *********************************

    # *********************************
    # node_psi辞書を作成して、node.psiにセットする
    # *********************************
    node_psi_dict_In4Dm = {}  # node_psi辞書を定義 # Inbound for Demand
    node_psi_dict_In4Sp = {}  # node_psi辞書を定義 # Inbound for Supply

    # rootからtree nodeをinbound4demand=preorder順に検索 node_psi辞書をmake
    node_psi_dict_In4Dm = make_psi_space_dict(
        root_node_inbound, node_psi_dict_In4Dm, plan_range
    )
    node_psi_dict_In4Sp = make_psi_space_dict(
        root_node_inbound, node_psi_dict_In4Sp, plan_range
    )

    # ***********************************
    # set_dict2tree
    # ***********************************
    # rootからtreeをinbound4demand=preorder順に検索 node_psi辞書をnodeにset
    set_dict2tree_In4Dm(root_node_inbound, node_psi_dict_In4Dm)
    set_dict2tree_In4Sp(root_node_inbound, node_psi_dict_In4Sp)

    # ***********************************
    # ここで、outboundとinboundを接続している
    # ***********************************
    connect_outbound2inbound(root_node_outbound, root_node_inbound)

    # S2P
    node_psi_dict_In4Dm = calc_bwd_inbound_all_si2p(
        root_node_inbound, node_psi_dict_In4Dm
    )

    # *************************
    # PS2Iで在庫を算定
    # *************************

    # calc_all_psi2i4demand_postorder(root_node_inbound, node_psi_dict_In4Dm)

    # *********************************
    # make visualise data for 3D bar graph
    # *********************************
    #    show_inbound_demand( root_node_inbound )

    # **********************************
    # leveling_inbound
    # **********************************
    # To Be defined

    # ********************************
    # bullwhip visualise
    # ********************************

    #    visualise_I_bullwhip4supply(root_node_outbound, 'out_supply_bullwhip.html')

    #    visualise_I_bullwhip4demand(root_node_inbound, 'in_demand_bullwhip.html')

    # ********************************
    # end2end supply chain accumed plan
    # ********************************

    # STOP
    # visualise_e2e_supply_chain_plan(root_node_outbound, root_node_inbound)



#    # ********************************
#    # map_psi_lots2df
#    # ********************************
#
#    # set_dataframe(root_node_outbound, root_node_inbound)
#
#    psi_lots = []  # 空リストを持ってtreeの中に入る
#    df_outbound_demand_plan = map_psi_lots2df(root_node_outbound, "demand", psi_lots)
#
#    psi_lots = []  # 空リストを持ってtreeの中に入る
#    df_outbound_supply_plan = map_psi_lots2df(root_node_outbound, "supply", psi_lots)
#
#    psi_lots = []  # 空リストを持ってtreeの中に入る
#    df_inbound_demand_plan = map_psi_lots2df(root_node_inbound, "demand", psi_lots)
#
#    psi_lots = []  # 空リストを持ってtreeの中に入る
#    df_inbound_supply_plan = map_psi_lots2df(root_node_inbound, "supply", psi_lots)
#
#    #print('df_outbound_supply_plan',df_outbound_supply_plan)
#
#    ####columns = ["node_name","week","s-co-i-p","step_no","lot_id"]
#
#    # pd.set_option('display.max_rows', 1500)
#    #
#    #print('df_outbound_supply_plan',df_outbound_supply_plan[df_outbound_supply_plan['node_name']=="CAN_I"])
#
#    # 2.making dataframe with pandas
#    #   df_outbound_demand_plan
#    #   df_outbound_supply_plan => push&pull control
#    #   df_inbound_demand_plan
#    #   df_inbound_supply_plan
#
#    # ************************
#    # show flow curve 流動曲線グラフ
#    # ************************
#
#    # df_init = df_inbound_demand_plan
#    # df_init = df_outbound_demand_plan
#
#    df_init = df_outbound_supply_plan
#
#    # node_show = "Platform"
#
#    node_show = "JPN"
#    # node_show = "HAM"
#    # node_show = "MUC"
#    # node_show = "MUC_D"
#
#    # node_show = "SHA_D"
#    # node_show = "SHA"
#    # node_show = "CAN_I"
#
#    # ************************
#    # 流動曲線はSTOP
#    # ************************
#    show_flow_curve( df_init, node_show )



    print_tree_dfs(root_node_outbound, depth=0)

    print_tree_bfs(root_node_outbound)

    print("end of process")


if __name__ == "__main__":
    main()
