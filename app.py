from flask import Flask, request, json, Response
from flask_restful import Api, Resource
from flask_cors import CORS
import re
import pymongo
import datetime
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model


app = Flask(__name__)
api = Api(app)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
client = pymongo.MongoClient("mongodb+srv://Saahil:FXOVdWdoxMtSKp1f@cluster0.flbld.mongodb.net/noradb?retryWrites=true&w=majority")
db = client.noradb


class Chat(Resource):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.intents = json.loads(open('static/intents.json').read())
        self.words = pickle.load(open('static/words.pkl', 'rb'))
        self.classes = pickle.load(open('static/classes.pkl', 'rb'))
        self.model = load_model('static/nora_model.h5')

    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word) for word in sentence_words]
        return sentence_words

    def bag_of_words(self, sentence):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    def predict_class(self, sentence):
        bow = self.bag_of_words(sentence)
        res = self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': self.classes[r[0]], 'probability': str(r[1])})
        return return_list

    def get_response(self, intents_list, intents_json):
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = [i['tag'], random.choice(i['responses'])]
                break
        return result

    def post(self):
        content = request.json
        message = content['message']
        ints = self.predict_class(message)
        res = self.get_response(ints, self.intents)
        data = {'status': 'success', 'data': {'intent': res[0], 'response': res[1]}}
        return Response(json.dumps(data), status=200, mimetype='application/json')


class Authenticate(Resource):
    def post(self):
        try:
            content = request.json
            assert re.match("^[0-9]{10}$", content['phoneno'])
            assert re.match("^[0-9]{4}$", content['otp'])
            result = db.users.find_one({
                "phoneno": content['phoneno'],
                "otp": content['otp']
            })
            user = {
                'phoneno': result['phoneno'],
                'firstName': result['firstName'],
                'middleName': result['middleName'],
                'lastName': result['lastName'],
                'age': result['age'],
                'gender': result['gender'],
                'address': result['address']
            }
            if result:
                data = {'status': 'success', 'data': {'task': 'authenticate', 'user': user, 'message': 'User is authenticated'}}
                print(data)
                return Response(json.dumps(data), status=200, mimetype='application/json')
            else:
                data = {'status': 'failure', 'data': {'task': 'authenticate', 'message': 'User is not authenticated'}}
                return Response(json.dumps(data), status=403, mimetype='application/json')
        except AssertionError:
            data = {'status': 'failure', 'data': { 'task': 'authenticate', 'message': 'Invalid data'}}
            return Response(json.dumps(data), status=400, mimetype='application/json')


class AddWithdraw(Resource):
    def post(self):
        try:
            content = request.json
            assert re.match("^[0-9]{10}$", content['phoneno'])
            assert isinstance(content['amount'], float)
            check_otp = db.users.find_one({
                "phoneno": content['phoneno'],
                "otp": content['otp']
            })
            if check_otp:
                if check_otp['balance']<content['amount']:
                    data = {'status': 'failure', 'data': {'task': 'add_withdraw', 'message': 'Balance not available'}}
                    return Response(json.dumps(data), status=400, mimetype='application/json')
                db.withdraws.insert_one({
                            "phoneno": content['phoneno'],
                            "amount": float(content['amount']),
                            "date": datetime.datetime.now(),
                            "purpose": content['purpose']
                        })
                db.users.update_one(
                    {"phoneno": content['phoneno']},
                    {"$inc": {"balance": -1 * content['amount']}}
                )
                data = {'status': 'success', 'data': {'task': 'add_withdraw', 'message': 'Data inserted to database'}}
                return Response(json.dumps(data), status=201, mimetype='application/json')
            else:
                data = {'status': 'failure', 'data': {'task': 'add_withdraw', 'message': 'User not verified'}}
                return Response(json.dumps(data), status=403, mimetype='application/json')
        except AssertionError:
            data = {'status': 'failure', 'data': {'task': 'add_withdraw', 'message': 'Invalid data'}}
            return Response(json.dumps(data), status=400, mimetype='application/json')


class AddDeposit(Resource):
    def post(self):
        try:
            content = request.json
            assert re.match("^[0-9]{10}$", content['phoneno'])
            assert isinstance(content['amount'], float)
            check_otp = db.users.find_one({
                "phoneno": content['phoneno'],
                "otp": content['otp']
            })
            if check_otp:
                db.deposits.insert_one({
                            "phoneno": content['phoneno'],
                            "amount": float(content['amount']),
                            "date": datetime.datetime.now(),
                            "purpose": content['purpose']
                        })
                db.users.update_one(
                    {"phoneno": content['phoneno']},
                    {"$inc": {"balance": content['amount']}}
                )
                data = {'status': 'success','data': {'task': 'add_deposit', 'message': 'Data inserted to database'}}
                return Response(json.dumps(data), status=201, mimetype='application/json')
            else:
                data = {'status': 'failure','data': {'task': 'add_deposit', 'message': 'User not verified'}}
                return Response(json.dumps(data), status=403, mimetype='application/json')
        except AssertionError:
            data = {'status': 'failure', 'data': {'task': 'add_deposit', 'message': 'Invalid data'}}
            return Response(json.dumps(data), status=400, mimetype='application/json')


class TransferBalance(Resource):
    def post(self):
        try:
            content = request.json
            assert re.match("^[0-9]{10}$", content['phoneno'])
            assert isinstance(content['amount'], float)
            assert re.match("^[0-9]{10}$", content['transferto'])

            check_otp = db.users.find_one({
                    "phoneno": content['phoneno'],
                    "otp": content['otp']
                })

            check_receiver = db.users.find_one({
                "phoneno": content['transferto'],
            })
            
            if check_otp and check_receiver:
                # withdraw from transferer
                if check_otp['balance']<content['amount']:
                    data = {'status': 'failure', 'data': {'task': 'transfer_balance', 'message': 'Balance not available'}}
                    return Response(json.dumps(data), status=400, mimetype='application/json')
                db.withdraws.insert_one({
                                "phoneno": content['phoneno'],
                                "amount": float(content['amount']),
                                "date": datetime.datetime.now(),
                                "purpose": content['purpose']
                            })
                db.users.update_one(
                    {"phoneno": content['phoneno']},
                    {"$inc": {"balance": -1 * content['amount']}}
                )

                # deposit to the transfered to
                db.deposits.insert_one({
                                "phoneno": content['transferto'],
                                "amount": float(content['amount']),
                                "date": datetime.datetime.now(),
                                "purpose": content['purpose']
                            })
                db.users.update_one(
                    {"phoneno": content['transferto']},
                    {"$inc": {"balance": content['amount']}}
                )

                data = {'status': 'success','data': {'task': 'transfer_balance', 'message': 'Data inserted to database'}}
                return Response(json.dumps(data), status=201, mimetype='application/json')
            else:
                data = {'status': 'failure','data': {'task': 'transfer_balance', 'message': 'User not verified'}}
                return Response(json.dumps(data), status=403, mimetype='application/json')
        
        except AssertionError:
            data = {'status': 'failure', 'data': {'task': 'transfer_balance', 'message': 'Invalid data'}}
            return Response(json.dumps(data), status=400, mimetype='application/json')


class CheckBalance(Resource):
    def post(self):
        try:
            content = request.json
            assert re.match("^[0-9]{10}$", content['phoneno'])
            result = db.users.find_one({
                "phoneno": content['phoneno']
            })
            data = {'status': 'success', 'data': {'task': 'check_balance', 'balance': result['balance'], 'message': 'Balance retreived'}}
            return Response(json.dumps(data), status=200, mimetype='application/json')

        except AssertionError:
            data = {'status': 'failure', 'data': {'task': 'check_balance', 'balance': None, 'message': 'Invalid data'}}
            return Response(json.dumps(data), status=400, mimetype='application/json')


class SaveChatLogs(Resource):
    def post(self):
        try:
            content = request.json
            db.chat_logs.insert(content)
            data = {'status': 'success', 'data': {'task': 'save_chat_logs', 'message': 'Chat log saved'}}
            return Response(json.dumps(data), status=200, mimetype='application/json')
        except:
            data = {'status': 'failure', 'data': {'task': 'save_chat_logs', 'message': 'Sorry an error occured'}}
            return Response(json.dumps(data), status=500, mimetype='application/json')


api.add_resource(Chat, "/chat")
api.add_resource(Authenticate, "/authenticate")
api.add_resource(AddWithdraw, "/add_withdraw")
api.add_resource(AddDeposit, "/add_deposit")
api.add_resource(TransferBalance, "/transfer_balance")
api.add_resource(CheckBalance, "/check_balance")
api.add_resource(SaveChatLogs, "/save_chat_logs")


if __name__ == '__main__':
    app.run()
