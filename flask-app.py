# run this flask app using torchrun --nproc_per_node MP flask-app.py

import flask
import string
import subprocess
from example import setup_model_parallel, load


app = flask.Flask(__name__)


# this runs when app starts

local_rank, world_size = setup_model_parallel()
if local_rank > 0:
    sys.stdout = open(os.devnull, "w")
gen_global = load(
    ckpt_dir = "../weights/7B",
    tokenizer_path = "../weights/tokenizer.model",
    local_rank = local_rank,
    world_size = world_size,
    max_seq_len = 512,
    max_batch_size = 32
)

@app.route("/flask-inference/", methods = ["POST"])
def flask_inference_no_batching():

	print("received POST request", flush=True)

	global gen_global

	req_data = flask.request.json
	print("request data", req_data, flush=True)

	print(req_data["apikey"], req_data["prompt"], flush = True) # for testing

	result = gen_global.generate(
        prompt, max_gen_len=256, temperature=0.8, top_p=0.95
    )
	print("result from model: ", result)

	res_data = {
		"apikey": req_data["apikey"],
		"result": result
	}

	return flask.jsonify(res_data)
