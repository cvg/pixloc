from pathlib import Path
import argparse
import http.server

PORT = 8000


class HttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        '': 'application/octet-stream',
        '.manifest': 'text/cache-manifest',
        '.html': 'text/html',
        '.png': 'image/png',
        '.jpg': 'image/jpg',
        '.svg':	'image/svg+xml',
        '.css':	'text/css',
        '.js': 'text/javascript',
        '.wasm': 'application/wasm',
        '.json': 'application/json',
        '.xml': 'application/xml',
    }


parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=PORT,
                    help='Server local port.')
args = parser.parse_args()
port = args.port

httpd = http.server.HTTPServer(('localhost', port), HttpRequestHandler)

try:
    relpath = Path(__file__).parent.resolve().relative_to(Path.cwd())
    print(f'Open the viewer at http://localhost:{port}/{relpath}/viewer.html')
    httpd.serve_forever()
except KeyboardInterrupt:
    pass
