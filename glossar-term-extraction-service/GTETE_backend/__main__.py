#!/usr/bin/env python3

import connexion
from flask_cors import CORS
import os
import GTETE_backend.encoder as encoder

dirname = os.path.dirname(__file__)


def main():
    app = connexion.App(__name__, specification_dir='swagger')
    app.app.json_encoder = encoder.JSONEncoder
    app.add_api('swagger.yaml', arguments={'title': 'Glossar Term Extraction'})
    CORS(app.app)
    app.run(host='0.0.0.0', port=8080)
    # app.run(host='localhost', port=8080)


if __name__ == '__main__':
    main()
