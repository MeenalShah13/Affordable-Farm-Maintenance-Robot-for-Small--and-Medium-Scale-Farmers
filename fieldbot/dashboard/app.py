"""
laptop_dashboard/app.py

A minimal Flask server that serves the single-page dashboard.
All data fetching is done in the browser using the Firebase JavaScript SDK —
this server only needs to serve the HTML file.

Run on your laptop:
    cd laptop_dashboard
    pip install flask
    python app.py

Then open:  http://localhost:8080
"""

from flask import Flask, render_template

app = Flask(__name__, template_folder="templates")


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    print("=" * 55)
    print("  FieldBot Dashboard  →  http://localhost:8080")
    print("=" * 55)
    app.run(host="0.0.0.0", port=8080, debug=True)
