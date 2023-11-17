from flask import Flask

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/imgs'

from libs import routes
