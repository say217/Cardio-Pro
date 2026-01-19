from app import create_app
import os

app = create_app()

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = app.config['DEBUG']
    app.run(host='0.0.0.0', port=port, debug=False)
else:

    gunicorn_app = app
    
    
    
    
    
    
