import os
import sqlite3
from flask import Flask, request, session, g, redirect, url_for, abort, \
    render_template, flash

app = Flask(__name__)  # create the application instance :)
# app.config.from_object(__name__)  # load config from this file , flaskr.py

# Load default config and override config from an environment variable
# app.config.update(dict(
#     DATABASE=os.path.join(app.root_path, 'flaskr.db'),
#     SECRET_KEY='development key',
#     USERNAME='admin',
#     PASSWORD='default'
# ))
# app.config.from_envvar('FLASKR_SETTINGS', silent=True)


# def connect_db():
#     """Connects to the specific database."""
#     rv = sqlite3.connect(app.config['DATABASE'])
#     rv.row_factory = sqlite3.Row
#     return rv


@app.route('/')
def show_entries():
    return 'Now it is time to start working on the templates. As you may have noticed, if you make requests with'


@app.route('/image_txt', methods=['POST', 'GET'])
def image_txt():
    return 'Now it is time to start working on the templates. As you may have noticed, if you make requests with ' \
           'the app running, you will get an exception that Flask cannot find the templates. The templates are using ' \
           'Jinja2 syntax and have autoescaping enabled by default. This means that unless you mark a value in the' \
           ' code with Markup or with the |safe filter in the template, Jinja2 will ensure that special characters' \
           ' such as < or > are escaped with their XML equivalents.'


app.run()
