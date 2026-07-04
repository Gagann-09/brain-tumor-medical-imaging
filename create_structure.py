import os

workspace = r"c:\Users\gagan\Downloads\PROJECTS\arm-gan-platform"

# Define the file content map
files_content = {
    # Backend
    "backend/main.py": """from fastapi import FastAPI

app = FastAPI(title="ARMT-GAN Platform API", version="1.0.0")

@app.get("/")
def read_root():
    return {"message": "Welcome to ARMT-GAN Platform API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
""",
    
    "backend/requirements.txt": """fastapi>=0.110.0
uvicorn>=0.28.0
pydantic>=2.6.0
sqlalchemy>=2.0.0
alembic>=1.13.0
python-dotenv>=1.0.0
python-multipart>=0.0.9
psycopg2-binary>=2.9.9
""",

    "backend/pyproject.toml": """[tool.poetry]
name = "armt-gan-backend"
version = "0.1.0"
description = "Backend for ARMT-GAN platform"
authors = ["Your Name <you@example.com>"]

[tool.poetry.dependencies]
python = "^3.10"
fastapi = "^0.110.0"
uvicorn = "^0.28.0"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
""",

    "backend/alembic.ini": """# A generic, single database configuration.

[alembic]
script_location = alembic
sqlalchemy.url = postgresql://postgres:postgres@localhost:5432/armt_gan

[post_write_hooks]

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARNING
handlers = console
qualname =

[logger_sqlalchemy]
level = WARNING
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
""",

    "backend/Dockerfile": """FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    libpq-dev \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
""",

    "backend/.env.example": """DATABASE_URL=postgresql://postgres:postgres@localhost:5432/armt_gan
SECRET_KEY=your_secret_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
ENVIRONMENT=development
""",

    # Frontend
    "frontend/package.json": """{
  "name": "armt-gan-frontend",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "react": "^18",
    "react-dom": "^18",
    "next": "14.1.0"
  },
  "devDependencies": {
    "typescript": "^5",
    "@types/node": "^20",
    "@types/react": "^18",
    "@types/react-dom": "^18",
    "eslint": "^8",
    "eslint-config-next": "14.1.0"
  }
}
""",

    "frontend/next.config.ts": """import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
};

export default nextConfig;
""",

    "frontend/tsconfig.json": """{
  "compilerOptions": {
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [
      {
        "name": "next"
      }
    ],
    "paths": {
      "@/*": ["./*"]
    }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
""",

    "frontend/middleware.ts": """import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function middleware(request: NextRequest) {
  return NextResponse.next();
}

export const config = {
  matcher: '/api/:path*',
};
""",

}

directories = [
    "frontend/public"
]

for d in directories:
    dir_path = os.path.join(workspace, d)
    os.makedirs(dir_path, exist_ok=True)
    print(f"Ensured directory exists: {d}")

for relative_path, content in files_content.items():
    file_path = os.path.join(workspace, relative_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Wrote file: {relative_path}")

print("Scaffold files setup completed.")
