import os
import requests

from flask import Flask, render_template, request, jsonify

import tensorflow as tf

from config import Config
from model import CaptionGenerator
from dataset import prepare_train_data, prepare_eval_data, prepare_test_data

app = Flask(__name__)

def download_file_from_google_drive(destination):
    URL = "https://public.boxcloud.com/d/1/b1!cThk862sN_mEYixwOAB-XR03H6HSHl7Of1enrson40a0Q86bGizmxgtAP9Rvg6uEIJb0cR43HGRb53_FCEFBUhJt0GG8hksOO-2Cqx-PN-CIm_c7LLvlVmbLDQrGPG2ck9GKmp19vm-2UqSKCPYL2K_zcBmP9h6SECnzGusb3VP0tSB_qgSkIhtE3TeVei3A1_jb4y9fPSHXBR1awbfELQlF8gZu9ElRgID_6zaA2yhAMrA59xHztdK1BKKW9BWUC5QqeHo3xewgXo-E_Cms5sYyM4aYeKd9cUFRGiEW41YNWHybCpRlxFJpYrikC72M_AsN-gC2CfPIvFpgE21cJFqA_vQYHO26L4QmvsYr0RMUzbTrWhO9c_I07ZTssV8InN-LwZNHAcj8prpPWovbR_pRX4OvB1eJalINAt4R8ZzLEdaenylhICK86Pi7-piEaiFbjE81LPu23NNHSO1BBDk8s2H_euMNxp5_agYQ5ee44hakye_CTRI8hGXzWG4EJdldOd_EmIjIIYNweFjtyFz_ye3koss24ZhfWOEEfkyrWAEBGQd5A_Q-to9dX6qr58vXfCCD0utjYG2UR3x6OOKh--YhrQ-O7umY42n-A90XToFYQj-LJIYD98BKe2ylLR2F8PwiF6kl_2zxoGHd8Ckzm-CSnZ_xviPPAAl4A__d0Bz5tKQil2woFFD0wkEfquSdqBt4WJIHbT8LX3GuB5fUGpqQnX3Dj2d-cJGmluTisQNisj1zQ0pCAnzQQMAWgpiH7T1biaGSGebXSegV3-rWWJVj6o8GO6hSaTMCErR_dPVAmrmqlNsg23FOv3u27Dl4jnKWHv-ZowWTsh8Y5JpFDQW2-09wmgdnpAvHeUbgRTdSgQn2uFJB4HQOQaM6YjcAHPFvpgKcgckS17ldzuQQK4rw6__rxh3Lw0nY5NX5ge-KdkWaq_yCoqsPxCv7QISXvHIeF07uEBNEOm9Q0VKlsqflUz69Vlsp0sM1tgdpCEu0tfC-avubD17pxUrQhvVA47NSC52AMsBIZG8RR6ql1Z0HXr2xtYgYUE3KonAY-ANLkPTubjem-P2z7p-PARJHrKqu97--zPgdiERQSzMmgCCCOHYBFeYbIeXmb9-TNTlbcsuCS8_NDC18jvBcvTpeoUkiTnHfVV8M9wyaMbZJwVaj2NXT0LdzfPvcIJkwHgtNk0zcz_oFdRZwj76emYd_9vXYqMkggvU0yXrkfRnQjVO0DsCK248k36CccSyWJOcaollzIdid3ivu4FMMAgCdF8E./download"

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
    if not os.path.exists(destination):
        download_file_from_google_drive(destination)

# setup_model()
config = Config()
config.beam_size = 3
config.phase = 'test'
config.train_cnn = False

sess = tf.Session()
model = CaptionGenerator(config)
model.load(sess)
tf.get_default_graph().finalize()

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    f = request.files['file']
    f.save(os.path.join('./test/images', f.filename))
    
    data, vocabulary = prepare_test_data(config)
    captions = model.test(sess, data, vocabulary)
    
    os.remove(os.path.join('./test/images', f.filename))
    return jsonify({'result': captions[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0',  port=80, debug=True)
