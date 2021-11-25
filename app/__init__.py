from flask import Flask, render_template, send_from_directory
# from app.middleware import PrefixMiddleware
import os
import socket

class PrefixMiddleware(object):

    def __init__(self, app, voc=True):
        self.app = app
        if voc:
            myip = self.get_myip()
            mytoken = os.getenv("VOC_PROXY_TOKEN")
            self.prefix = f'/hostip/{myip}:5000/vocproxy/{mytoken}'
        else:
            self.prefix = ''

    def __call__(self, environ, start_response):
        print(environ['PATH_INFO'], self.prefix)
        environ['SCRIPT_NAME'] = self.prefix
        return self.app(environ, start_response)

    def get_myip(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 53))
        return s.getsockname()[0]

# application = Flask(__name__)
application = Flask(__name__, static_folder="src", template_folder='src')

# set voc=False if you run on local computer
application.wsgi_app = PrefixMiddleware(application.wsgi_app, voc=False)

@application.route('/')
@application.route('/index')
def index():
    # return render_template('src\index.html')
    return send_from_directory("src", 'index.html')

if __name__ == "__main__":
 application.run(host="0.0.0.0", port=8080, debug=True)
 

# from flask import Flask
# # from flask_bootstrap import Bootstrap 
# from app.middleware import PrefixMiddleware

# application = Flask(__name__)
# # bootstrap = Bootstrap(application)

# # set voc=False if you run on local computer
# application.wsgi_app = PrefixMiddleware(application.wsgi_app, voc=False)

# from app import routes