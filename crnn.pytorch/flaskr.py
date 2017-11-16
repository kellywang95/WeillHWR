import os
import sqlite3
from flask import Flask, request, session, g, redirect, url_for, abort, \
    render_template, flash

import run_for_given_file

app = Flask(__name__)  # create the application instance :)


@app.route('/')
def show_entries():
    return 'Now it is time to start working on the templates. As you may have noticed, if you make requests with'


@app.route('/image_txt', methods=['POST'])
def image_txt():
    print(request.form)
    print(request)
    if 'index' not in request.form:
        raise Exception("Send me the Index!")

    return run_for_given_file.extract_result(request.form['index'])


app.run()
