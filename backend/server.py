# Import flask and datetime module for showing date and time
from flask import Flask, jsonify, request,current_app,make_response
import datetime
from flask_cors import CORS
from pyzomato import General
# Initializing flask app
app = Flask(__name__)
CORS(app)

# Route for seeing a data
@app.route('/api', methods=['POST'])
def post_object():
    data = request.get_json() ##converint from JSON to DICT
    print('result',data)
    return General(data)

 
# Running app
if __name__ == '__main__':
	app.run(debug=True)

