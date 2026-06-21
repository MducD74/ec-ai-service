# middleware/logging_middleware.py
import re
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from logger import http_log, app_log

SKIP_REGEX = re.compile(
    r".*\.(js|css|png|jpg|jpeg|gif|svg|ico|woff|woff2|ttf|map)$"
)


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        url = str(request.url.path)

        if SKIP_REGEX.match(url):
            return await call_next(request)

        start_time = time.time()

        try:
            response = await call_next(request)

            process_time = round((time.time() - start_time) * 1000, 2)

            client_ip = request.client.host if request.client else "-"
            method = request.method
            status_code = response.status_code
            user_agent = request.headers.get("user-agent", "-")

            http_log.info(
                f'{client_ip} "{method} {url}" {status_code} {process_time}ms "{user_agent}"'
            )

            return response

        except Exception as e:
            process_time = round((time.time() - start_time) * 1000, 2)

            app_log.exception(
                f'Unhandled error: {request.method} {url} after {process_time}ms. Error: {str(e)}'
            )

            raise