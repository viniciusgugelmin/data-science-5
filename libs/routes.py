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
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support

from libs import app

from io import BytesIO
import base64

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

x_treino, x_teste, y_treino, y_teste = train_test_split(trainers_escala, targets, test_size=0.3, random_state=0)


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
            matplotlib.use('Agg')

            fig, ax = plt.subplots(figsize=(10, 10))
            ax = sns.set_style("darkgrid")

            sns.set()

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

        if args.get('v2') and args.get('v2') == 'confusion-matrix' and args.get('v'):
            type = args.get('v', 'default')

            if type == 'default':
                return render_template('home.html')

            matplotlib.use('Agg')

            fig, ax = plt.subplots(figsize=(10, 10))
            ax = sns.set_style("darkgrid")
            sns.set()

            params = {}

            if type == 'decision-tree':
                params = {
                    'criterion': 'entropy',
                    'random_state': 0,
                    'max_depth': 4
                }

                args_criterion = args.get('criterion')
                args_random_state = args.get('random_state')
                args_max_depth = args.get('max_depth')

                if args_criterion:
                    params['criterion'] = args_criterion

                if args_random_state:
                    params['random_state'] = int(args_random_state)

                if args_max_depth:
                    params['max_depth'] = int(args_max_depth)

                classifier = DecisionTreeClassifier(**params)
            elif type == 'random-forest':
                params = {
                    'n_estimators': 100,
                    'criterion': 'entropy',
                    'random_state': 0
                }

                args_n_estimators = args.get('n_estimators')
                args_criterion = args.get('criterion')
                args_random_state = args.get('random_state')

                if args_n_estimators:
                    params['n_estimators'] = int(args_n_estimators)

                if args_criterion:
                    params['criterion'] = args_criterion

                if args_random_state:
                    params['random_state'] = int(args_random_state)

                classifier = RandomForestClassifier(**params)
            elif type == 'svc':
                params = {
                    'kernel': 'rbf',
                    'random_state': 0
                }

                args_kernel = args.get('kernel')
                args_random_state = args.get('random_state')

                if args_kernel:
                    params['kernel'] = args_kernel

                if args_random_state:
                    params['random_state'] = int(args_random_state)

                classifier = SVC(**params)
            elif type == 'knn':
                params = {
                    'n_neighbors': 5,
                    'metric': 'minkowski',
                    'p': 2
                }

                args_n_neighbors = args.get('n_neighbors')
                args_metric = args.get('metric')
                args_p = args.get('p')

                if args_n_neighbors:
                    params['n_neighbors'] = int(args_n_neighbors)

                if args_metric:
                    params['metric'] = args_metric

                if args_p:
                    params['p'] = int(args_p)

                classifier = KNeighborsClassifier(**params)

            _classifier = classifier.fit(x_treino, y_treino)
            y_pred = _classifier.predict(x_teste)

            cm = confusion_matrix(y_teste, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')

            accuracy = _classifier.score(x_teste, y_teste)
            accuracy = "{:.2f}".format(accuracy * 100)

            precision, recall, fscore, support = precision_recall_fscore_support(y_teste, y_pred, average='macro')

            macro_avg = "{:.2f}".format(fscore * 100)

            path = os.path.abspath(os.path.dirname(__file__))
            path2 = app.config['UPLOAD_FOLDER']
            path3 = f'{path}/{path2}/confusion_matrix.png'

            plt.savefig(path3)

            return render_template('home.html', confusion_matrix_img_d3=True, accuracy=accuracy, macro_avg=macro_avg, confusion_matrix=True, params=params)

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


@app.route('/api/decision-tree', methods=['GET'])
def show_decision_tree():
    if request.method == 'GET':
        return redirect(url_for('homepage', v="decision-tree", v2="confusion-matrix"))

    return False


@app.route('/api/random-forest', methods=['GET'])
def show_random_forest():
    if request.method == 'GET':
        return redirect(url_for('homepage', v="random-forest", v2="confusion-matrix"))

    return False


@app.route('/api/svc', methods=['GET'])
def show_svc():
    if request.method == 'GET':
        return redirect(url_for('homepage', v="svc", v2="confusion-matrix"))

    return False


@app.route('/api/knn', methods=['GET'])
def show_knn():
    if request.method == 'GET':
        return redirect(url_for('homepage', v="knn", v2="confusion-matrix"))

    return False
