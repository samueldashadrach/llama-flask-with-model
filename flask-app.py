import flask
import string
import random

app = flask.Flask(__name__)


@app.route("/")
def hello():
	print("flask server is running", flush=True)
	return "flask server is running"

@app.route("/flask-inference/", methods = ["POST"])
def flask_inference_no_batching():

	print("received POST request", flush=True)

	req_data = flask.request.json
	print("request data", req_data, flush=True) # for testing

	apikey, prompt = req_data["apikey"], req_data["prompt"]
	print(req_data["apikey"], req_data["prompt"], flush = True) # for testing

	# write to file named "prompt"
	tempname = "".join(random.choices(string.ascii_uppercase, k=20))
	try:
		with open(tempname, "w") as f_prompt_temp:
			f_prompt_temp.write(prompt)
			f_prompt_temp.close()
			os.rename(tempname, "prompt")

	except IOError:
		print("prompt could not be written!!!")

	# wait for file named "result" to be created, then read and destroy "result" file
	result_found = False
	while not result_found:
		try:
			with open("result", "r") as f_result:
				result = f_result.read()
				f_result.close()
				os.remove("result")
				result_found = True
		except IOError:
			pass

	print("result from model: ", result)

	res_data = {
		"apikey": req_data["apikey"],
		"result": result
	}

	return flask.jsonify(res_data)
