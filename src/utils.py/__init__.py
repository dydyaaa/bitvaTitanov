import json
import os

from flask import Flask, jsonify


def create_app(test_mode=True):
    
    app = Flask(__name__)
    
    settings = 'settings_test' if test_mode else 'settings'
    config_path =os.path.abspath(os.path.join(os.path.dirname(__file__), '..', f'{settings}.json'))
    
    with open(config_path) as config_file:
        config = json.load(config_file)
        app.config.update(config)
        
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'message': 'Not Found'}), 404
    
    @app.errorhandler(Exception)
    def internal_server_error(error):
        app.logger.error(f'{error}')
        return jsonify({'message': 'Internal Server Error'}), 500
    
    return app