from flask import Flask, render_template, request, jsonify
from collections import Counter
import pandas as pd
import matplotlib
import os

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

app = Flask(__name__)
file_name = '/Users/jinnie/Desktop/25FA/CBB 6340/js4872/project/data/NSDUH_2023_Tab.txt'
df = pd.read_csv(file_name, sep="\t", dtype=str)
ever_use_dict = {'Cigarette': 'CIGEVER', 'Alcohol': 'ALCEVER', 'Marijuana': 'MJEVER',
                 'Cocaine': 'COCEVER', 'Heroin': 'HEREVER'}
ever_value_code = {'1': 'Yes', '2': 'No'}

# lowercase input -> canonical key
ever_use_lc = {k.lower(): k for k in ever_use_dict.keys()}

STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')
os.makedirs(STATIC_DIR, exist_ok=True)


@app.route('/api/counts', methods=['GET'])
def api_counts():
    # e.g. query param: ?substance=Alcohol
    substance = request.args.get('substance', type=str)
    if not substance:
        return jsonify({'error': 'missing substance parameter', 'available': list(ever_use_dict.keys())}), 400

    # case-insensitive match
    canonical = ever_use_lc.get(substance.strip().lower())
    if canonical is None:
        return jsonify({'error': f"unrecognized substance '{substance}'", 'available': list(ever_use_dict.keys())}), 400

    col = ever_use_dict[canonical]
    raw_counts = df[col].value_counts().to_dict()
    counts = {ever_value_code.get(str(k), str(k)): int(v) for k, v in raw_counts.items() if str(k) in ever_value_code}
    yes = counts.get('Yes', 0)
    no = counts.get('No', 0)
    total = yes + no
    pct_yes = round(100 * yes / total, 2) if total else 0.0

    return jsonify({'substance': substance, 'yes': yes, 'no': no, 'total': total, 'pct_yes': pct_yes})


def test_api_counts(substance='Cigarette'):
    with app.test_client() as client:
        resp = client.get('/api/counts', query_string={'substance': substance})
        print('Requesting /api/counts?substance=' + substance)
        print('Status code:', resp.status_code)
        try:
            print('JSON response:', resp.get_json())
        except Exception:
            print('Response data:', resp.data)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    usertext = request.form["usertext"]
    if not usertext:
        return render_template("analyze.html", analysis="No substance provided.", usertext="")

    canonical = ever_use_lc.get(usertext.strip().lower())
    if canonical is None:
        msg = f"Unrecognized substance '{usertext}'. Available: {', '.join(sorted(ever_use_dict.keys()))}"
        return render_template("analyze.html", analysis=msg, usertext=usertext)

    col = ever_use_dict[canonical]
    counts = df[col].value_counts().to_dict()
    # only keep '1' and '2' keys
    counts = {ever_value_code[str(k)]: v for k, v in counts.items() if str(k) in ever_value_code}
    result = ""
    for item, count in counts.items():
        if item == 'Yes':
            result += f"The number of people who have EVER used {usertext} is {count}.\n"
        elif item == 'No':
            result += f"The number of people who have NEVER used {usertext} is {count}.\n"
    
    # Generate pie chart
    labels = list(counts.keys())
    sizes = [int(v) for v in counts.values()]
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.title(f"Proportion of {usertext} Use")
    plt.legend(loc="best")
    save_path = os.path.join(STATIC_DIR, 'pie_chart.png')
    plt.savefig(save_path)
    plt.close()

    return render_template("analyze.html", analysis=result, usertext=usertext)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'test_api':
        substance = sys.argv[2] if len(sys.argv) > 2 else 'Cigarette'
        test_api_counts(substance)
    else:
        app.run()