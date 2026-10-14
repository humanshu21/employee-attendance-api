# Employee Attendance and Analytics API

FastAPI and MongoDB implementation of the HROne attendance contract. All API code is in app/main.py. MongoDB 6.0 or newer is required.

Install dependencies with python -m pip install -r requirements.txt. Set MONGO_URI and MONGO_DB in the environment, then run uvicorn app.main:app --port 8000 from this directory.

The app creates its indexes at startup. GET /health checks the MongoDB connection.
