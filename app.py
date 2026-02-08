from flask import Flask, jsonify, render_template
import sqlite3

app = Flask(__name__)

def get_latest_cumulative():
    conn = sqlite3.connect('benguet_crowd.db')
    c = conn.cursor()
    # Fetch the highest total count recorded for the road
    c.execute("""SELECT road_name, total_count FROM realtime_crowd 
                 WHERE id IN (SELECT MAX(id) FROM realtime_crowd GROUP BY road_name)""")
    rows = c.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}

@app.route('/api/crowd')
def crowd_api():
    return jsonify(get_latest_cumulative())

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)