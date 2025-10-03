#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def test_encoding():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
        <title>Encoding Test</title>
    </head>
    <body>
        <h1>🌸 Encoding Test</h1>
        <p>If you can see the flower emoji above, encoding is working!</p>
        <p>Test characters: 🌸 🤖 📊 🎯 ✅ ❌</p>
    </body>
    </html>
    """
    return HTMLResponse(
        content=html_content, 
        media_type="text/html; charset=utf-8",
        headers={"Content-Type": "text/html; charset=utf-8"}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)