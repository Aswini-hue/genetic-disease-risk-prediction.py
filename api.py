from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "Project":"Genetic Data Compression and Pattern Visualization System",
        "Status":"Running"
    })

if __name__ == "__main__":
    app.run(debug=True)