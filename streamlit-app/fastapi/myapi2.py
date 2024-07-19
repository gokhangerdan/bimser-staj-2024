from fastapi import FastAPI
from typing import Optional


app = FastAPI()

@app.get("/blog")
def index(limit = 10, published: bool = True, sort: Optional[str] = None):
    # only get 10 published blogs
    if published:
        return {"data": f"{limit} published blogs form the db"}
    else:
        return {"data": f"{limit} blogs form the db"}
@app.get("/blog/{id}")
def show(id):
    return {"data": id} 