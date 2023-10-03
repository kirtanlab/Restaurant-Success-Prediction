# Import flask and datetime module for showing date and time
from flask import Flask, jsonify, request,current_app,make_response
import datetime
from flask_cors import CORS
from pyzomato import General
# from flask_sqlalchemy import SQLAlchemy
import os
# Initializing flask app
app = Flask(__name__)
CORS(app)


# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:admin@localhost/ZOMATO'
# app.config['CORS_HEADERS'] = 'Content-Type'

# app.app_context().push()
# db = SQLAlchemy(app)

# class Rest_Type_Table(db.Model):
#     id = db.Column(db.Integer, primary_key=True,autoincrement=True)
#     Rest_Type = db.Column(db.String(200), unique=True, nullable=False)
#     Value = db.Column(db.String(80), nullable=False) 
#     def __init__(self, Rest_Type,Value):
#         self.Rest_Type = Rest_Type
#         self.Value = Value
# class Cus_Type_Table(db.Model):
#     id = db.Column(db.Integer, primary_key=True,autoincrement=True)
#     Cus_Type = db.Column(db.String(200), unique=True, nullable=False)
#     Value = db.Column(db.String(80), nullable=False) 
#     def __init__(self, Cus_Type,Value):
#         self.Cus_Type = Cus_Type
#         self.Value = Value 
# class Location_Table(db.Model):
#     id = db.Column(db.Integer, primary_key=True,autoincrement=True)
#     Location = db.Column(db.String(200), unique=True, nullable=False)
#     Value = db.Column(db.String(80), nullable=False) 
#     def __init__(self, Location,Value):
#         self.Location = Location
#         self.Value = Value 
# db.create_all()

@app.route("/",methods=["POST"])
def index():
    response_data = {'message': 'Success!'}
    status_code = 200  
    return response_data, status_code



# Route for seeing a data
@app.route('/api', methods=['POST'])
def post_object():
    data = request.get_json() ##converint from JSON to DICT
    # print('result',data)
    return General(data)

 
# Running app
if __name__ == '__main__':
	app.run(debug=True)

