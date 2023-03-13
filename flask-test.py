import flask

app = flask.Flask(__name__)

@app.route("/")
def hello():
	print("flask server is running", flush=True)
	return("flask server is running")