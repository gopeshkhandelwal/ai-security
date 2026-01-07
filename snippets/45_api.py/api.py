import flask
import pymysql
from flask import request, jsonify
from flaskext.mysql import MySQL
from flask_jwt import JWT, jwt_required
from werkzeug.security import safe_str_cmp
from flasgger import Swagger


class User(object):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

    def __str__(self):
        return "User(id='%s')" % self.id


users = [
    User(1, 'root', 'root'),
    User(2, 'admin', 'admin'),
]

username_table = {u.username: u for u in users}
userid_table = {u.id: u for u in users}

def authenticate(username, password):
    user = username_table.get(username, None)
    if user and safe_str_cmp(user.password.encode('utf-8'), password.encode('utf-8')):
        return user

def identity(payload):
    user_id = payload['identity']
    return userid_table.get(user_id, None)


app = flask.Flask(__name__)
app.config["DEBUG"] = True
app.config['SECRET_KEY'] = 'super-secret'
# jwt = JWT(app, authenticate, identity)

"""

curl -d '{"username": "admin", "password":"admin"}' -H 'Content-Type: application/json' http://127.0.0.1:5000/auth -v
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZGVudGl0eSI6MiwiaWF0IjoxNjAxNzM2MjAyLCJuYmYiOjE2MDE3MzYyMDIsImV4cCI6MTYwMTczNjUwMn0.ywAwJEayJOOZ2s1Kk7y40n_v3rRX8H-2TYSM1hYXxRA"
}

curl http://127.0.0.1:5000/api/v1/resources/todos -v
curl -H 'Content-Type: application/json' -H 'Authorization: JWT eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZGVudGl0eSI6MiwiaWF0IjoxNjAxNzUxMDY2LCJuYmYiOjE2MDE3NTEwNjYsImV4cCI6MTYwMTc1MTM2Nn0._5KmYgv2Alpp7OzDp_SvpZg7y5N7Lw6vV6Lm_o7aqgc' http://127.0.0.1:5000/api/v1/resources/todos


curl -d '{"username": "admin", "password":"admin"}' -H 'Content-Type: application/json' http://127.0.0.1:5000/auth -v

curl http://127.0.0.1:5000/api/v1/resources/todos -v
"""

swagger = Swagger(app)

mysql = MySQL()

# MySQL configurations
app.config['MYSQL_DATABASE_HOST'] = 'db'
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = 'root'
app.config['MYSQL_DATABASE_DB'] = 'vulnapi'

mysql.init_app(app)

con = mysql.connect()

@app.route('/', methods=['GET'])
def home():
    return "<h1>Practical DevSecOps TODOs API</h1>"

@app.route('/api/v1/resources/todos', methods=['GET'])
# @jwt_required()
def get_all_todos():
    """Returns a list of todo item
    ---
    produces:
    - "application/json"
    responses:
      200:
        description: "list of tasks"
        schema:
          type: "array"
          items:
            $ref: "#/definitions/Task"
      400:
        description: "Invalid ID supplied"
      404:
        description: "todo item not found"
    """
    cursor = con.cursor(pymysql.cursors.DictCursor)
    cursor.execute("SELECT * from todos")
    todos = cursor.fetchall()
    cursor.close()
    return jsonify(todos), 200

@app.route('/api/v1/resources/todos/<string:id>', methods=['GET'])
def get_todo(id):
    """Returns a todo item
    ---
    produces:
    - "application/json"
    parameters:
    - name: "id"
      in: "path"
      description: "ID of todo item to return"
      required: true
      type: "integer"
      format: "int64"
      default: all
    responses:
      200:
        description: "successful operation"
        schema:
          $ref: "#/definitions/Task"
      400:
        description: "Invalid ID supplied"
      404:
        description: "todo item not found"
    definitions:
        Task:
            type: "object"
            properties:
              id:
                type: "integer"
                format: "int64"
              user:
                type: "string"
              todo:
                type: "string"
              date:
                type: "string"
                format: date
    """
    """
    if 'id' in request.args:
        id = request.args['id']
    else:
        error = {"Error": "No id found. Please specify an id."}
        return jsonify(error), 404
    """
    cursor = con.cursor(pymysql.cursors.DictCursor)
    cursor.execute("SELECT * from todos where id="+str(id))
    todo = cursor.fetchone()
    cursor.close()
    return jsonify(todo), 200

@app.route('/api/v1/resources/todos', methods=['POST'])
def add_todo():
    """ Add a new task to the store
    ---
    consumes:
    - "application/json"
    produces:
    - "application/json"
    parameters:
    - in: "body"
      name: "body"
      description: "Task object that needs to be added to the store"
      required: true
      schema:
        $ref: "#/definitions/Task"
    responses:
      201:
        description: "Task added"
      405:
        description: "Invalid input"
    """
    todo = request.get_json()
    print(todo)
    cursor = con.cursor(pymysql.cursors.DictCursor)
    query = "INSERT INTO todos(user, todo, date) VALUES(%s, %s, %s)"
    values = (todo['user'], todo['todo'], todo['date'])
    cursor.execute(query, values)
    con.commit()
    cursor.close()
    return jsonify(todo), 201

@app.route('/api/v1/resources/todos/<string:id>', methods=['PUT'])
def update_todo(id):
    """ Update an existing todo item
    ---
    consumes:
    - "application/json"
    produces:
    - "application/json"
    parameters:
    - name: "id"
      in: "path"
      description: "ID of todo item to return"
      required: true
      type: "integer"
      format: "int64"
    - in: "body"
      name: "body"
      description: "Task object that needs to be added to the store"
      required: true
      schema:
        $ref: "#/definitions/Task"
    responses:
      400:
        description: "Invalid ID supplied"
      404:
        description: "Task not found"
      405:
        description: "Validation exception"
    """
    todo = request.get_json()
    # print(todo['task'])
    cursor = con.cursor(pymysql.cursors.DictCursor)

    # UPDATE customers SET address = 'Canyon 123',  user=imran WHERE address = 'Valley 345'
    query = "Update todos set user=%s, todo=%s, date=%s where id=%s"
    data = (todo['user'], todo['todo'], todo['date'], str(id))
    cursor.execute(query, data)
    con.commit()

    cursor.execute("SELECT * from todos where id=%s",str(id))
    todo = cursor.fetchone()
    cursor.close()
    return jsonify(todo), 200

@app.route('/api/v1/resources/todos/<int:id>', methods=['DELETE'])
def delete_todo(id):
    """summary: "Deletes a task"
    ---
    produces:
    - "application/json"
    parameters:
    - name: "id"
      in: "path"
      description: "Task id to delete"
      required: true
      type: "integer"
      format: "int64"
    responses:
      200:
        description: "Task deleted"
      400:
        description: "Invalid ID supplied"
      404:
        description: "Task not found"
    """
    cursor = con.cursor(pymysql.cursors.DictCursor)
    cursor.execute("DELETE FROM todos WHERE id=%s",str(id))
    print(dir(cursor))
    con.commit()
    cursor.close()
    message = "Id: " + str(id) + " Deleted"
    json_message = {"message": message}
    return jsonify(json_message), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0')