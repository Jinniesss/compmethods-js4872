from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import io
import base64

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report

app = Flask(__name__)

# Path to preprocessed CSV (must be in the same folder as this server.py)
DATA_PATH = '/Users/jinnie/Desktop/25FA/CBB 6340/js4872/project/preprocessed.csv'


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close()
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('ascii')
    return img_b64


@app.route('/', methods=['GET'])
def index():
    # default values
    defaults = {'k': 5, 'pca_n': 64}

    # load data and produce interactive 2D PCA plot (Plotly) with selectable kingdoms
    df = load_data()

    # determine codon columns and ensure numeric
    metadata_numeric = ['SpeciesID', 'Ncodons', 'Kingdom', 'DNAtype', 'SpeciesName', 'Taxon_group', 'PC1', 'PC2', 'PC3']
    codon_cols = [c for c in df.columns if c not in metadata_numeric]

    # compute PC1/PC2 if not present
    if not ({'PC1', 'PC2'}.issubset(set(df.columns))):
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        X = df[codon_cols].astype(float).values
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        pca = PCA(n_components=2)
        Xp = pca.fit_transform(Xs)
        df['PC1'] = Xp[:, 0]
        df['PC2'] = Xp[:, 1]

    kingdoms_available = sorted(df['Taxon_group'].unique())
    # get selected kingdoms from query string (GET), default all
    selected = request.args.getlist('kingdom')
    if not selected:
        selected = kingdoms_available

    # prepare DataFrame for plotting
    plot_df = df[df['Taxon_group'].isin(selected)].copy()

    # color mapping consistent with notebook
    color_map = { 'Plant': 'green', 'Animal': 'orange', 'Bacteria': 'red', 'Virus': 'blue' }

    try:
        # compute fixed square axis limits from the full dataset so selections don't change coordinates
        x_all = df['PC1'].astype(float).values
        y_all = df['PC2'].astype(float).values
        x_min, x_max = float(x_all.min()), float(x_all.max())
        y_min, y_max = float(y_all.min()), float(y_all.max())
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        half_span = max(0.5 * (x_max - x_min), 0.5 * (y_max - y_min))
        # add small padding
        pad = 0.05 * half_span if half_span > 0 else 0.5
        half_span = half_span + pad
        x_range = [cx - half_span, cx + half_span]
        y_range = [cy - half_span, cy + half_span]

        fig = px.scatter(plot_df, x='PC1', y='PC2', color='Taxon_group', hover_data=['SpeciesName'],
                         color_discrete_map=color_map, title='PCA of Codon Usage: PC1 vs PC2')
        # increase figure size while keeping square aspect
        fig.update_layout(legend_title_text='Kingdom', width=900, height=900, margin=dict(l=40, r=40, t=60, b=40))
        fig.update_xaxes(range=x_range)
        fig.update_yaxes(range=y_range, scaleanchor="x", scaleratio=1)
        pca2_plot_html = fig.to_html(include_plotlyjs='cdn', full_html=False)
    except Exception:
        pca2_plot_html = None

    return render_template('index.html', defaults=defaults, pca2_plot_html=pca2_plot_html,
                           kingdoms=kingdoms_available, selected=selected)


@app.route('/knn', methods=['GET'])
def knn():
    # simple page to host the kNN form (separate from PCA/index)
    defaults = {'k': 5, 'pca_n': 64}
    return render_template('knn.html', defaults=defaults)


@app.route('/run', methods=['POST'])
def run():
    # read form
    k = int(request.form.get('k', 5))
    pca_n = int(request.form.get('pca_n', 64))

    # compute results via helper and render result template
    results = compute_knn_results(k, pca_n)
    # create confusion matrix image for the result page
    cm_img = plot_confusion_matrix(np.array(results['cm']), labels=results['labels'])

    context = {
        'k': k,
        'pca_n': pca_n,
        'accuracy': f"{results['accuracy']:.3f}",
        'balanced_accuracy': f"{results['balanced_accuracy']:.3f}",
        'cv_bal_mean': f"{results['cv_bal_mean']:.3f}" if results.get('cv_bal_mean') is not None else 'N/A',
        'cv_bal_std': f"{results['cv_bal_std']:.3f}" if results.get('cv_bal_std') is not None else 'N/A',
        'report': results['report'],
        'cm_img': cm_img
    }

    return render_template('result.html', **context)


def compute_knn_results(k, pca_n):
    """Compute kNN predictions and return structured results (JSON-serializable)."""
    df = load_data()

    metadata_numeric = ['SpeciesID', 'Ncodons', 'Kingdom', 'DNAtype', 'SpeciesName', 'Taxon_group', 'PC1', 'PC2', 'PC3']
    codon_cols = [c for c in df.columns if c not in metadata_numeric]

    X_raw = df[codon_cols].values
    y = df['Taxon_group'].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # split
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_enc, test_size=0.25, random_state=42, stratify=y_enc)

    # build pipeline with requested PCA and k
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=pca_n)),
        ('knn', KNeighborsClassifier(n_neighbors=k, weights='distance'))
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    # structured classification report
    try:
        report = classification_report(y_test, y_pred, target_names=list(le.classes_), output_dict=True, zero_division=0)
    except Exception:
        report = {}

    cm = confusion_matrix(y_test, y_pred, normalize='true')

    # also compute cross-validated balanced accuracy on whole data for reference
    try:
        cv_bal = cross_val_score(pipe, X_raw, y_enc, cv=5, scoring='balanced_accuracy', n_jobs=-1)
        cv_bal_mean = float(np.mean(cv_bal))
        cv_bal_std = float(np.std(cv_bal))
    except Exception:
        cv_bal_mean = None
        cv_bal_std = None

    results = {
        'k': k,
        'pca_n': pca_n,
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'cv_bal_mean': cv_bal_mean,
        'cv_bal_std': cv_bal_std,
        'report': report,
        'cm': cm.tolist(),
        'labels': list(le.classes_)
    }

    return results


@app.route('/api/predict', methods=['GET'])
def api_predict():
    """
    Example: /api/predict?k=5&pca_n=30
    """
    k = int(request.args.get('k', 5))
    pca_n = int(request.args.get('pca_n', 64))
    results = compute_knn_results(k, pca_n)
    
    return jsonify(results)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8888, debug=True)
