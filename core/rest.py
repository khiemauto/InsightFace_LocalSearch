from fastapi.param_functions import Body, Form

from fastapi import FastAPI, File, UploadFile, status, Query
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse, StreamingResponse

from core import support
from pydantic import BaseModel


class FaceRecogAPI(FastAPI):
    def __init__(self, title: str = "FaceRecogAPI"):
        super().__init__(title=title)
        self.image_type = (".jpg", ".jpeg", ".png", ".bmp")

        @self.get('/')
        async def home():
            """
            Home page
            """
            return HTMLResponse("<h1>Face Recognition API</h1><br/><a href='/docs'>Try api now!</a>", status_code=status.HTTP_200_OK)

        @self.post("/add_sayname")
        async def add_sayname(say_name: dict):
            """
            Add name to TTS
            say_name: Customer name
            """
            try:
                print(say_name)
                name = say_name["name"]
                if name is None or name=="":
                    return PlainTextResponse(f"Name is incorrect string value", status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)

                support.add_sayname_queue(str(name))
                return PlainTextResponse(f"Add sayname ok!", status_code=status.HTTP_200_OK)
            except:
                return PlainTextResponse(f"Error", status_code=status.HTTP_417_EXPECTATION_FAILED)