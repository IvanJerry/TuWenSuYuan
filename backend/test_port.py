from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)                     # 允许跨域

@app.route("/api/hello", methods=["POST"])
def hello():
    name = request.json.get("name", "World")
    return jsonify(msg=f"你好, {name}!")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)