import os

from flask import Flask, render_template, request, jsonify

import tensorflow as tf

from config import Config
from model import CaptionGenerator
from dataset import prepare_train_data, prepare_eval_data, prepare_test_data

app = Flask(__name__)

def download_file_from_google_drive(destination):
    URL = "https://public.boxcloud.com/d/1/b1!bTmGt8PhTS3I3aFG_slS876XMtLmwhsCDkMlNAOGDvZ7FtB7pe7l_vPP6O5poVxfL18bPQoibqDZlndeBUuNFUmN5f4as-MTva-KdGAPylLSRJhmCFEZQWE8XMC-geWQCzzG8_73LI9n6QFLAOztbCyksT6sEezZUbuqkMSpeVDBTsuq6AGfg5msiZi3TdiEb-6nuFwm6aJjaH7JbgLluKZzTKGS2oK-EUtJV9bRZFQTGYN267ouChN3_N_GlY0kXesowvA145OKiT9UVCBy_fGHz27nZKQYtECTCqhufkbW_6xin9AvvxdySzMxGra7dF5bz8PJi2ZC1djgq8k0NUkwTUyeWpNWjhDG5uM6ZFaOsnh2pwpm4CLJm6mXsVHGTqG2ijra3PeHikpB2X9veyronksBjsKdz9nHhuPH-QTED1ufWwXR0GOQpERzkSDs0W0hn9EQ3FZ0Xa6D9EcXaHWEPNS9CASCm2z2E9vUAvmDP3PN-XNMV1Kx-FkXQjNkR7fARvITGdQaejOOkIJBX91xWNZFmN3Cx0ZqFjmSsxNxsB76hsAxVCaT05R6vAUaif3mGwXDAVWILIbrpGS58o1FEmfedET5Dp3qF15r8IbXj_X0PvyciKPcES3GJAW9m_1aPXrYbwT0MCmDMC9Pc3je_T0Oz4W4CgtVzw_0ZVgbshaKr2nWFVPegl1ce5zpcpJdJzs5fnmBgNzeBj8pg8gp6v_vPcujnPJewObUdmJoJSDnpsTgg0xRfm0CkUEnWLmiBC8IBY7GmzvJZ-wmv9_YDgujTtOWiJ6JU1OPRM-448Bzrs0in_7mQIft-ovsnZYx3bO35Jz6SfyBGfyIzirQMx7K8aY1Du9mhiHhARPFEkvy5OPlEHGL2dBbwvxukDYdRKJX-QXq4jpq8eTajV6GpiMb-EByQr7r8pZAzABFxt9w1U-RyR9C8XYlb9-3sv2rskYEXD8hf34TmEKq82K1S6oD-INcUtBKZCw4F45xxKiMvg7fqp5woLfqqvgkM3z7jf5lJcUW9s5DG6OXcVIBZDrmbIIOIF3Dmc7BWAUj87koXRtgQIu217bwVQK_5-nkqZXjyfqnwwou0LFq2yDvLv0ejrHr4U6A2FFo_ra3EIff6gFrWpZsnaEkDyyMVBfKr_f2rYXEArPDus1bw_i9wihJoCInjOK26smqKmXyELUmYVLLJt-q1Z1G0u3PHAHxnPJbCaHJY2Sv7nQUMbCfsX8BPqlTUQbGYUv0-pdMM80eVh1_EJoApAUOOQ__Tsw./download"

    session = requests.Session()

    response = session.get(URL, stream = True)

    save_response_content(response, destination)

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def setup_model():
    destination = 'models/model.npy'
    download_file_from_google_drive(destination)

setup_model() 

@app.route('/')
def index():
   render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    f = request.files['file']
    f.save(os.path.join('./test/images', f.filename))

    config = Config()
    config.beam_size = 3
    config.phase = 'test'
    config.train_cnn = False

    with tf.Session() as sess:
        data, vocabulary = prepare_test_data(config)
        model = CaptionGenerator(config)
        model.load(sess)
        tf.get_default_graph().finalize()
        captions = model.test(sess, data, vocabulary)
    
    os.remove(os.path.join('./test/images', f.filename))
    return jsonify({'captions': captions})

if __name__ == '__main__':
    app.run(host='0.0.0.0',  port=8080, debug=True)
