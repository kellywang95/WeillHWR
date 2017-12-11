import os
import sqlite3
from flask import Flask, request, session, g, redirect, url_for, abort, \
    render_template, flash, jsonify

import run_for_given_file

from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)  # create the application instance :)


@app.route('/')
def show_entries():
    return render_template('index.html')


@app.route('/login', methods=['GET'])
def initial():
    username, password = request.args.get('username'), request.args.get('password')
    if username is not None:
        if run_for_given_file.check_creds(username, password):
            return redirect(url_for('show_entries'))
    return render_template('login.html')


@app.route('/image_txt', methods=['POST'])
def image_txt():
    if 'index' not in request.form:
        raise Exception("Send me the Index!")

    return run_for_given_file.extract_result(request.form['index'])


@app.route('/search_txt', methods=['POST'])
def search_txt():
    if 'keyword' not in request.form:
        raise Exception("Send me the keywords!")

    return jsonify(run_for_given_file.get_most_relevant(request.form['keyword'], 3))


@app.route('/single_image', methods=['POST'])
def single_image():
    if 'image' not in request.form:
        raise Exception("Send me the image for this word!")

    k = Image.open(BytesIO(base64.b64decode(request.form['image'].split(",")[1])))
    k.save("abcd.png")
    k.close()

    return run_for_given_file.extract_for_image('abcd.png')

app.run()
