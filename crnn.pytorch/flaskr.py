import os
import sqlite3
from flask import Flask, request, session, g, redirect, url_for, abort, \
    render_template, flash

import run_for_given_file

app = Flask(__name__)  # create the application instance :)


@app.route('/')
def show_entries():
    return render_template('index.html')

# @app.route('/login.html')
# def showLogin():
#     return render_template('login.html')
#
# @app.route('/kelly')
# def show_home():
#     return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != app.config['USERNAME']:
            error = 'Invalid username'
        elif request.form['password'] != app.config['PASSWORD']:
            error = 'Invalid password'
        else:
            session['logged_in'] = True
            flash('You were logged in')
            return redirect(url_for('show_entries'))
    return render_template('login.html', error=error)


@app.route('/image_txt', methods=['POST'])
def image_txt():
    print(request.form)
    print(request)
    if 'index' not in request.form:
        raise Exception("Send me the Index!")

    return run_for_given_file.extract_result(request.form['index'])


app.run()
