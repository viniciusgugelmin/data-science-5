import os

from flask import render_template, redirect, url_for, request

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from libs import app

from io import BytesIO
import base64

# base configuration
matplotlib.use('Agg')

fig, ax = plt.subplots(figsize=(8, 8))
ax = sns.set_style("darkgrid")

sns.set()

# load data
path = os.path.abspath(os.path.dirname(__file__))
csv_url = f'{path}/data/dataset.csv'

df = pd.read_csv(csv_url, sep=';')
df = df.sort_values(by=['rating'], ascending=False)

# standardize the data
le = LabelEncoder()

df_sample = df.copy()

df_sample["rating"] = le.fit_transform(df_sample.rating.values)

df_sample = df_sample.drop(['name', 'agent_1', 'gun1_name', 'headshot_percent', 'win_percent'], axis=1)

df_sample_min = df_sample.sample(frac=0.10)

trainers = df_sample_min.iloc[:, 1:9].values
targets = df_sample_min.iloc[:, 0].values

trainers_escala = StandardScaler().fit_transform(trainers)

@app.route('/', methods=['GET'])
def homepage():
    if request.method == 'GET':
        args = request.args

        if args.get('v') and args.get('v') == 'dataset':
            items_per_page = 30
            number_of_pages = int(np.ceil(df.shape[0] / items_per_page))
            page = int(args.get('page', 1))

            if page > number_of_pages:
                page = 1

            start = (page - 1) * items_per_page
            end = start + items_per_page

            df_to_show = df.iloc[start:end].to_dict('records')

            pagination = {
                'first': 1,
                'has_previous': page > 1,
                'has_next': page < number_of_pages,
                'active': page,
                'previous': page - 1,
                'next': page + 1,
                'last': number_of_pages
            }

            return render_template('home.html', dataset=df_to_show, pagination=pagination)

        if args.get('v') and args.get('v') == 'correlation':
            df_exec = df_sample.copy()
            df_exec = df_exec.drop(['rating'], axis=1)
            corr = df_exec.corr()
            cmap = sns.diverging_palette(220, 20, as_cmap=True)
            sns.heatmap(corr, cmap=cmap, annot=True)

            canvas = FigureCanvas(plt.figure())
            img = BytesIO()
            fig.savefig(img)
            img.seek(0)

            img_base64 = base64.b64encode(img.getvalue()).decode('utf8')

            return render_template('home.html', correlation_img=img_base64)

        return render_template('home.html')


@app.route('/api/dataset', methods=['GET'])
def show_dataset():
    args = request.args

    if request.method == 'GET':
        return redirect(url_for('homepage', v="dataset", page=args.get('page', 1)))

    return False


@app.route('/api/correlation', methods=['GET'])
def show_correlation():
    args = request.args

    if request.method == 'GET':
        return redirect(url_for('homepage', v="correlation"))

    return False
